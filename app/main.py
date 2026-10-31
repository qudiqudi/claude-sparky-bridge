"""claude-sparky-bridge: an OpenAI-compatible /v1/chat/completions in front of `claude -p`.

SparkyFitness (service type openai_compatible, URL http://claude-sparky-bridge:8080/v1)
sends its chat turns, food-photo estimates and label scans here. Each request
becomes one `claude -p` process on your Claude subscription.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hmac
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect

from .claude_runner import ClaudeError, ClaudeResult, run_claude
from .config import cfg, resolve_model
from .translate import BadRequest, Plan, TooLarge, json_text, plan_request, tool_calls_from

logging.basicConfig(level=os.environ.get("BRIDGE_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("bridge")

app = FastAPI(title="claude-sparky-bridge", docs_url=None, redoc_url=None, openapi_url=None)

KEEPALIVE_S = 10
_counter_lock = asyncio.Lock()


# --- parallel claude processes ----------------------------------------------

_slots = asyncio.Semaphore(cfg.max_concurrent)
_running = 0
_waiting = 0


@asynccontextmanager
async def _claude_slot() -> AsyncIterator[None]:
    """One of BRIDGE_MAX_CONCURRENT claude processes. Over BRIDGE_MAX_QUEUED waiters: 429 at once."""
    global _running, _waiting
    if _waiting >= cfg.max_queued:
        raise ClaudeError(f"too many requests are waiting ({cfg.max_queued})", 429)
    _waiting += 1
    try:
        # A cancelled waiter leaves here through CancelledError, without a slot.
        await _slots.acquire()
    finally:
        _waiting -= 1
    _running += 1
    try:
        yield
    finally:
        _running -= 1
        _slots.release()


def queue_state() -> dict[str, int]:
    return {"running": _running, "waiting": _waiting}


# --- daily limit -----------------------------------------------------------


def _read_counter() -> dict[str, Any]:
    try:
        with open(cfg.counter_file) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


async def _take_call() -> None:
    today = dt.date.today().isoformat()
    async with _counter_lock:
        data = _read_counter()
        count = data.get("count", 0) if data.get("date") == today else 0
        if count >= cfg.daily_limit:
            raise ClaudeError(f"daily limit of {cfg.daily_limit} calls is reached", 429)
        os.makedirs(os.path.dirname(cfg.counter_file) or ".", exist_ok=True)
        tmp = cfg.counter_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"date": today, "count": count + 1}, fh)
        os.replace(tmp, cfg.counter_file)


def calls_today() -> int:
    data = _read_counter()
    return data.get("count", 0) if data.get("date") == dt.date.today().isoformat() else 0


# --- helpers ---------------------------------------------------------------


def _error(status: int, message: str, kind: str = "bridge_error") -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind}})


async def _stop(task: asyncio.Task) -> None:
    """Cancel a running _execute and wait until claude is gone."""
    if not task.done():
        task.cancel()
    try:
        await task
    except BaseException:
        pass


def _failure(exc: BaseException) -> tuple[int, str, str]:
    """(status, message, type) for an exception from _execute."""
    if isinstance(exc, TooLarge):
        return 413, str(exc), "invalid_request_error"
    if isinstance(exc, BadRequest):
        return 400, str(exc), "invalid_request_error"
    if isinstance(exc, ClaudeError):
        log.warning("error: %s", str(exc)[:160])
        return exc.status, str(exc), "bridge_error"
    log.exception("unexpected error")
    return 500, "internal bridge error", "bridge_error"


def _authorized(request: Request) -> bool:
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    return bool(cfg.bridge_token) and hmac.compare_digest(token.encode(), cfg.bridge_token.encode())


async def _execute(plan: Plan, model: str) -> tuple[str | None, list[dict[str, Any]], dict[str, int]]:
    """Run one plan. Returns (content, tool_calls, usage)."""
    if plan.kind == "probe":
        return "OK", [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async with _claude_slot():
        await _take_call()
        if plan.kind == "classifier":
            result = await _timed(plan, cfg.classifier_model, run_claude(
                plan.prompt,
                model=cfg.classifier_model,
                system_prompt=plan.system_prompt,
                blocks=plan.content_blocks(),
                timeout_s=cfg.classifier_timeout_s,
                effort="low",
            ))
        else:
            result = await _timed(plan, model, _run_plan(plan, model))
        if plan.kind == "tools":
            return await _tools_reply(plan, model, result)

    if plan.kind == "schema":
        return json.dumps(result.structured, ensure_ascii=False), [], result.usage
    text = result.text
    if plan.json_mode and text:
        text = json_text(text)
    if not text or not text.strip():
        raise ClaudeError("claude gave an empty answer")
    return text, [], result.usage


async def _timed(plan: Plan, model: str, run: Any) -> ClaudeResult:
    """Await one claude run and write the log line for it."""
    t0 = time.monotonic()
    result: ClaudeResult = await run
    log.info(
        "kind=%s model=%s prompt_bytes=%d images=%d ms=%d duration_api_ms=%s turns=%s native_calls=%d "
        "tokens_in=%d input_tokens=%d cache_creation_input_tokens=%d cache_read_input_tokens=%d tokens_out=%d",
        plan.kind,
        model,
        len(plan.prompt.encode()),
        len(plan.images),
        int((time.monotonic() - t0) * 1000),
        "-" if result.api_ms is None else result.api_ms,
        "intercepted" if result.native_calls is not None else result.num_turns,
        result.intercepted,
        result.usage["prompt_tokens"],
        result.counts.get("input_tokens", 0),
        result.counts.get("cache_creation_input_tokens", 0),
        result.counts.get("cache_read_input_tokens", 0),
        result.usage["completion_tokens"],
    )
    return result


def _structured_of(result: ClaudeResult) -> Any:
    if result.native_calls is not None:
        # claude called a caller tool natively; the runner checked the
        # arguments and stopped it there.
        return {"text": result.text or None, "tool_calls": result.native_calls}
    return result.structured


async def _tools_reply(plan: Plan, model: str, result: ClaudeResult) -> tuple[str | None, list[dict[str, Any]], dict[str, int]]:
    """The reply of a tools plan. Runs inside the claude slot.

    A tool call whose arguments do not match the schema of its tool never
    goes to the caller: SparkyFitness would refuse it, and the chat would
    show an error card. claude gets one more run that names the error, and
    a call that is still wrong after that is dropped.
    """
    text, calls, invalid = tool_calls_from(_structured_of(result), plan.tool_names, plan.tool_schemas)
    if invalid:
        log.warning("tool call does not match its schema, claude runs once more: %s", _invalid_text(invalid))
        await _take_call()
        again = replace(plan, prompt=plan.prompt + _retry_note(_structured_of(result), invalid))
        result = await _timed(again, model, _run_plan(again, model))
        text, calls, invalid = tool_calls_from(_structured_of(result), plan.tool_names, plan.tool_schemas)
        if invalid:
            log.warning("tool call does not match its schema, dropped: %s", _invalid_text(invalid))
            if not calls:
                raise ClaudeError("claude gave a tool call that does not match the schema of the tool")
    if not calls and not text:
        raise ClaudeError("claude gave neither text nor a known tool call")
    return text, calls, result.usage


def _invalid_text(invalid: list[tuple[str, str]]) -> str:
    return "; ".join(f"{name} at {error}" for name, error in invalid)


def _retry_note(structured: Any, invalid: list[tuple[str, str]]) -> str:
    """What claude gets to see of its refused reply: the calls and their errors, for the model only."""
    names = {name for name, _ in invalid}
    tried = [c for c in (structured.get("tool_calls") or []) if isinstance(c, dict) and c.get("name") in names]
    return (
        "\n\nYour previous reply was not accepted. These tool calls have arguments that do not match the JSON schema "
        f"of the tool (the position and the failed schema keyword are given): {_invalid_text(invalid)}.\n"
        f"The refused calls: {json.dumps(tried, ensure_ascii=False)}\n"
        "Write the reply again with arguments that match the schema in <tools>. Give arrays as JSON arrays and numbers as JSON numbers, not as strings."
    )


# The cache mark that the API accepted last: "1h", "5m" or None (no mark).
# claude sets its own cache marks, with a lifetime that depends on the login
# (1h seen with a setup-token, 5m with a local login). A mark with a longer
# lifetime must not come after one with a shorter lifetime, and the API takes
# at most four marks. The first chat after a start finds the working choice.
_cache_state: dict[str, str | None] = {"ttl": "1h"}


async def _run_plan(plan: Plan, model: str) -> ClaudeResult:
    """One claude run for a chat, schema or text plan.

    A tools plan gets a cache mark on its <tools> block (BRIDGE_CACHE_TOOLS).
    When the API refuses the mark, the run goes again with the other
    lifetime, then without a mark, and the bridge keeps the choice that works.
    """
    if not (cfg.cache_tools and plan.cache_prefix):
        return await _run_once(plan, model, None)
    first = _cache_state["ttl"]
    tries: list[str | None] = [first] if first is None else list(dict.fromkeys([first, "1h", "5m"])) + [None]
    for i, ttl in enumerate(tries):
        if i:
            await _take_call()
        try:
            result = await _run_once(plan, model, ttl)
        except ClaudeError as exc:
            if ttl is None or not (exc.from_result and _cache_mark_refused(str(exc))):
                raise
            log.warning("claude refused the cache mark (ttl=%s), next: %s: %s", ttl, tries[i + 1] or "no mark", str(exc)[:160])
            continue
        if _cache_state["ttl"] != ttl:
            log.info("cache mark is now %s", ttl or "off")
            _cache_state["ttl"] = ttl
        return result
    raise ClaudeError("no cache mark choice worked")  # not reached: the last try has no mark


def _cache_mark_refused(message: str) -> bool:
    """An API 400 about cache_control: too many marks, or the order of their lifetimes."""
    text = message.lower()
    return "cache_control" in text and "400" in text


async def _run_once(plan: Plan, model: str, cache_ttl: str | None) -> ClaudeResult:
    return await run_claude(
        plan.prompt,
        model=model,
        system_prompt=plan.system_prompt,
        json_schema=plan.json_schema,
        blocks=plan.content_blocks(cache_ttl=cache_ttl),
        tool_schemas=plan.tool_schemas if plan.kind == "tools" else None,
    )


def _completion(cid: str, model: str, content: str | None, calls: list[dict[str, Any]], usage: dict[str, int]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": usage,
    }


def _chunk(cid: str, model: str, delta: dict[str, Any], finish: str | None = None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream(request: Request, plan: Plan, model: str, echo_model: str, include_usage: bool) -> AsyncIterator[str]:
    cid = f"chatcmpl-{secrets.token_hex(12)}"
    task = asyncio.create_task(_execute(plan, model))
    try:
        yield _chunk(cid, echo_model, {"role": "assistant", "content": ""})
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_S)
            if done:
                break
            if await request.is_disconnected():
                await _stop(task)
                log.info("client went away, claude stopped")
                return
            # SSE comment: parsers ignore it, proxies see traffic.
            yield ": keep-alive\n\n"
        try:
            content, calls, usage = task.result()
        except Exception as exc:
            _, message, kind = _failure(exc)
            yield f"data: {json.dumps({'error': {'message': message, 'type': kind}})}\n\n"
            yield "data: [DONE]\n\n"
            return
        if content:
            yield _chunk(cid, echo_model, {"content": content})
        for i, call in enumerate(calls):
            yield _chunk(cid, echo_model, {"tool_calls": [{"index": i, **call}]})
        yield _chunk(cid, echo_model, {}, "tool_calls" if calls else "stop")
        if include_usage:
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": echo_model,
                "choices": [],
                "usage": usage,
            }
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await _stop(task)


# --- routes ----------------------------------------------------------------


async def _read_body(request: Request) -> bytes:
    """The raw body, at most cfg.max_body_bytes. Raises TooLarge or BadRequest."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > cfg.max_body_bytes:
                raise TooLarge(f"body is larger than {cfg.max_body_bytes} bytes")
        except ValueError as exc:
            raise BadRequest("content-length is not a number") from exc
    # A chunked body has no Content-Length: count while it arrives.
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > cfg.max_body_bytes:
            raise TooLarge(f"body is larger than {cfg.max_body_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "configured": cfg.configured,
        "calls_today": calls_today(),
        "daily_limit": cfg.daily_limit,
        **queue_state(),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not _authorized(request):
        return _error(401, "invalid bridge token", "authentication_error")
    if not cfg.configured:
        return _error(503, "bridge is not configured (CLAUDE_CODE_OAUTH_TOKEN / BRIDGE_TOKEN)")
    try:
        body = json.loads(await _read_body(request))
    except TooLarge as exc:
        return _error(413, str(exc), "invalid_request_error")
    except BadRequest as exc:
        return _error(400, str(exc), "invalid_request_error")
    except ClientDisconnect:
        return _error(499, "client closed request")
    except ValueError:
        # json.JSONDecodeError and UnicodeDecodeError
        return _error(400, "body is not JSON", "invalid_request_error")
    if not isinstance(body, dict):
        return _error(400, "body must be a JSON object", "invalid_request_error")

    try:
        plan = plan_request(body)
    except TooLarge as exc:
        return _error(413, str(exc), "invalid_request_error")
    except BadRequest as exc:
        return _error(400, str(exc), "invalid_request_error")
    except Exception:
        log.warning("malformed request body", exc_info=True)
        return _error(400, "malformed request body", "invalid_request_error")

    echo_model = body.get("model") if isinstance(body.get("model"), str) else cfg.default_model
    model = resolve_model(body.get("model"))

    if body.get("stream"):
        options = body.get("stream_options")
        include_usage = isinstance(options, dict) and bool(options.get("include_usage"))
        return StreamingResponse(
            _stream(request, plan, model, echo_model, include_usage),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    task = asyncio.create_task(_execute(plan, model))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_S)
            if done:
                break
            if await request.is_disconnected():
                log.info("client went away, claude stopped")
                return _error(499, "client closed request")
        content, calls, usage = task.result()
    except Exception as exc:
        return _error(*_failure(exc))
    finally:
        # Also runs when this handler is cancelled: claude must not run on.
        await _stop(task)
    return _completion(f"chatcmpl-{secrets.token_hex(12)}", echo_model, content, calls, usage)
