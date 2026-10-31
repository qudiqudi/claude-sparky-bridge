"""Runs the pinned `claude` binary headless, one process per request.

Isolation:
  * The environment holds only CLAUDE_CODE_OAUTH_TOKEN, HOME, PATH,
    DISABLE_AUTOUPDATER and TZ.
  * settings/settings.json denies every tool, and `--tools ""` gives claude
    none. Images are not written to disk: they travel inside the message.
  * The message (text and images) goes over stdin as one stream-json line,
    never as an argv element. Only the system prompt and the schema are argv.

Output is read as stream-json, line by line, while claude runs. The run ends
at the result event: claude is killed there, because its own shutdown after
the result takes close to a second and gives nothing. For a request with
caller tools, claude sometimes calls one of them as a native tool (which does
not exist in the process) instead of writing it into the StructuredOutput
reply. The runner takes such a call as the reply at once and kills the
process in the same way: waiting would only give "no such tool". Claude has
no schema for such a tool, so the runner repairs and checks the arguments
first (see translate.coerce_arguments); a call that still does not match
the schema of its tool is not taken, and claude continues: it then gets
"no such tool" and writes the call into the StructuredOutput reply.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .config import cfg
from .translate import argument_error, coerce_arguments

log = logging.getLogger("bridge.claude")

MAX_ARG_BYTES = 120_000
# One stdout line of claude; the structured reply travels in one line.
STDOUT_LINE_LIMIT = 32 * 1024 * 1024


USAGE_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


class ClaudeError(Exception):
    def __init__(self, message: str, status: int = 502, from_result: bool = False):
        super().__init__(message)
        self.status = status
        # True when the message is the error text of claude's result event.
        self.from_result = from_result


@dataclass
class ClaudeResult:
    text: str
    structured: Any
    usage: dict[str, int]  # OpenAI shape: prompt_tokens, completion_tokens, total_tokens
    duration_ms: int | None
    num_turns: int | None = None
    # Caller tools that claude called natively: [{"name", "arguments"}], in order.
    # None when the reply came from the result event.
    native_calls: list[dict[str, Any]] | None = None
    intercepted: int = 0  # number of native calls taken as the reply
    # The usage of the API, by USAGE_KEYS, and the API time of the result event.
    counts: dict[str, int] = field(default_factory=dict)
    api_ms: int | None = None


@dataclass
class _Stream:
    """What the stdout reader collected."""

    result: dict[str, Any] | None = None
    native_calls: list[dict[str, Any]] = field(default_factory=list)
    native_text: str = ""
    native_usage: dict[str, Any] = field(default_factory=dict)


def _counts(raw: Any) -> dict[str, int]:
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for key in USAGE_KEYS:
        value = raw.get(key)
        out[key] = int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
    return out


def _usage(counts: dict[str, int]) -> dict[str, int]:
    prompt_tokens = counts["input_tokens"] + counts["cache_read_input_tokens"] + counts["cache_creation_input_tokens"]
    completion_tokens = counts["output_tokens"]
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def build_command(
    *,
    model: str,
    system_prompt: str,
    json_schema: dict[str, Any] | None,
    effort: str | None = None,
) -> list[str]:
    command = [
        cfg.claude_bin,
        "-p",
        "--verbose",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--model",
        model,
        "--effort",
        effort or cfg.effort,
        "--system-prompt",
        system_prompt,
        "--settings",
        os.path.join(cfg.settings_dir, "settings.json"),
        "--strict-mcp-config",
        "--no-session-persistence",
        "--tools",
        "",
    ]
    if json_schema is not None:
        command += ["--json-schema", json.dumps(json_schema, separators=(",", ":"))]
    return command


def _stdin_line(blocks: list[dict[str, Any]]) -> bytes:
    message = {"type": "user", "message": {"role": "user", "content": blocks}}
    return (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")


async def run_claude(
    prompt: str,
    *,
    model: str,
    system_prompt: str,
    json_schema: dict[str, Any] | None = None,
    blocks: list[dict[str, Any]] | None = None,
    tool_schemas: dict[str, Any] | None = None,
    timeout_s: int | None = None,
    effort: str | None = None,
) -> ClaudeResult:
    """One claude process.

    `prompt` is the text (checked against BRIDGE_MAX_PROMPT_BYTES). `blocks`
    is the message content that goes to claude (text and image blocks); it
    defaults to one text block with the prompt. `tool_schemas` maps the name
    of each caller tool to its JSON schema: a native call of such a tool,
    with arguments that match the schema, ends the run at once.
    """
    if len(prompt.encode("utf-8")) > cfg.max_prompt_bytes:
        raise ClaudeError(f"prompt is larger than {cfg.max_prompt_bytes} bytes", 413)
    os.makedirs(cfg.home, exist_ok=True)
    env = {
        "CLAUDE_CODE_OAUTH_TOKEN": cfg.oauth_token,
        "HOME": cfg.home,
        "PATH": "/usr/bin:/bin",
        "DISABLE_AUTOUPDATER": "1",
        "TZ": os.environ.get("TZ", "UTC"),
    }
    command = build_command(model=model, system_prompt=system_prompt, json_schema=json_schema, effort=effort)
    # System prompt and schema travel as argv; Linux caps one argument at 128 KiB.
    if max(len(a.encode("utf-8")) for a in command) > MAX_ARG_BYTES:
        raise ClaudeError(f"system prompt or schema is larger than {MAX_ARG_BYTES} bytes", 413)
    data = _stdin_line(blocks if blocks is not None else [{"type": "text", "text": prompt}])

    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cfg.home,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=STDOUT_LINE_LIMIT,
        )
    except FileNotFoundError as exc:
        raise ClaudeError(f"claude binary not found: {cfg.claude_bin}", 500) from exc
    except OSError as exc:
        raise ClaudeError(f"claude could not start: {exc.strerror}", 500) from exc

    limit = timeout_s or cfg.timeout_s
    # stderr and stdin have their own tasks, so a full pipe can not block the run.
    stderr_task = asyncio.create_task(proc.stderr.read())
    writer_task = asyncio.create_task(_write_stdin(proc, data))
    try:
        stream = await asyncio.wait_for(_read_stdout(proc, tool_schemas or {}), timeout=limit)
    except asyncio.TimeoutError as exc:
        await _kill(proc)
        await _finish(writer_task, stderr_task)
        raise ClaudeError(f"claude did not answer in {limit} s", 504) from exc
    except asyncio.CancelledError:
        # The caller went away (SparkyFitness aborted the turn). Do not leave
        # the process running on the subscription.
        await asyncio.shield(_kill(proc))
        await asyncio.shield(_finish(writer_task, stderr_task))
        raise

    if stream.native_calls or stream.result is not None:
        # The reply is complete: stop claude at once. After a native call it
        # would chase a tool that does not exist; after the result event it
        # only shuts down. Its exit code is then ours and says nothing.
        await _kill(proc)
        await _finish(writer_task, stderr_task)
    if stream.native_calls:
        counts = _counts(stream.native_usage)
        return ClaudeResult(
            text=stream.native_text,
            structured=None,
            usage=_usage(counts),
            duration_ms=None,
            num_turns=None,
            native_calls=stream.native_calls,
            intercepted=len(stream.native_calls),
            counts=counts,
        )
    if stream.result is not None:
        return _from_result(stream.result, json_schema)

    # stdout ended without a result event: the exit code and stderr tell why.
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except asyncio.TimeoutError as exc:
        await _kill(proc)
        await _finish(writer_task, stderr_task)
        raise ClaudeError("claude did not exit after its output ended") from exc
    await _finish(writer_task)
    stderr = (await stderr_task).decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        log.warning("claude exit=%s reason=%s", proc.returncode, stderr[:160])
        raise ClaudeError(f"claude ended with exit code {proc.returncode}: {stderr[:300]}")
    raise ClaudeError(f"claude gave no result event: {stderr[:200]}")


def _from_result(envelope: dict[str, Any], json_schema: dict[str, Any] | None) -> ClaudeResult:
    if envelope.get("is_error"):
        raise ClaudeError(f"claude reported an error: {str(envelope.get('result'))[:300]}", from_result=True)
    if json_schema is not None and "structured_output" not in envelope:
        raise ClaudeError("claude gave no structured_output")
    counts = _counts(envelope.get("usage"))
    api_ms = envelope.get("duration_api_ms")
    return ClaudeResult(
        text=envelope.get("result") if isinstance(envelope.get("result"), str) else "",
        structured=envelope.get("structured_output"),
        usage=_usage(counts),
        duration_ms=envelope.get("duration_ms"),
        num_turns=envelope.get("num_turns"),
        counts=counts,
        api_ms=int(api_ms) if isinstance(api_ms, (int, float)) else None,
    )


async def _write_stdin(proc: asyncio.subprocess.Process, data: bytes) -> None:
    try:
        proc.stdin.write(data)
        await proc.stdin.drain()
        proc.stdin.close()
    except (BrokenPipeError, ConnectionResetError):
        # claude ended before it read the message; the exit code tells why.
        pass


async def _finish(*tasks: asyncio.Task) -> None:
    """Cancel helper tasks and wait for them; their results are not needed."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _read_stdout(proc: asyncio.subprocess.Process, tool_schemas: dict[str, Any]) -> _Stream:
    """Parse events until the result event, a valid native call of a caller tool, or the end of stdout."""
    stream = _Stream()
    while True:
        line = await proc.stdout.readline()
        if not line:
            return stream
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "result":
            stream.result = event
            return stream
        if kind == "assistant" and tool_schemas and _take_native_calls(event, tool_schemas, stream):
            return stream


def _take_native_calls(event: dict[str, Any], tool_schemas: dict[str, Any], stream: _Stream) -> bool:
    message = event.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), list):
        return False
    calls = []
    texts = []
    invalid = []
    for block in message["content"]:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and block.get("name") in tool_schemas:
            name = block["name"]
            arguments = coerce_arguments(block.get("input"), tool_schemas[name])
            error = argument_error(arguments, tool_schemas[name])
            if error is not None or not isinstance(arguments, dict):
                invalid.append(f"{name} at {error or '. (type)'}")
            calls.append({"name": name, "arguments": arguments})
        elif block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
    if not calls:
        # StructuredOutput or any other tool: let claude continue.
        return False
    if invalid:
        # The caller would refuse the call. claude gets "no such tool" for it
        # and writes the next try into the StructuredOutput reply.
        log.warning("native tool call does not match its schema, claude continues: %s", "; ".join(invalid))
        return False
    stream.native_calls = calls
    stream.native_text = "\n".join(t for t in texts if t.strip())
    stream.native_usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
    return True


async def _kill(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            return
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        log.warning("claude pid=%s did not exit after kill", proc.pid)
