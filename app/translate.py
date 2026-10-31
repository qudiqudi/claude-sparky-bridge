"""OpenAI Chat Completions request -> one `claude -p` prompt, and back.

`claude -p` takes one text prompt per process and has no native OpenAI tool
calls. The bridge therefore writes the whole conversation (with earlier tool
calls and tool results) as a transcript into the prompt, lists the tools of
the caller, and forces the reply into a JSON schema: either text or a list of
tool calls. SparkyFitness then executes its own tools and calls again.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import secrets
from dataclasses import dataclass, field
from typing import Any

import jsonschema
from jsonschema.exceptions import best_match
from jsonschema.validators import validator_for

from .config import cfg

PROBE_TEXT = "Reply with the single word: OK."
CLASSIFIER_MARKER = "determine which of the following health tracking domains"

BRIDGE_RULES = """\
You are the language model behind an OpenAI-compatible chat API.
The caller (the SparkyFitness app) executes the tools listed in <tools>. You do not execute them yourself and you have no other tools.
To use a tool, put it into "tool_calls" with "arguments" that match its JSON schema. You can call several tools in one reply.
Never say that a listed tool is unavailable. After a tool call, its result appears in the conversation as [tool result ...]; use it.
When no further tool is needed, put the answer for the user into "text" and set "tool_calls" to null.
When you call tools, set "text" to null or to one short sentence for the user.
Images are attached to the conversation.
When the system prompt tells you to call a tool for this kind of request, you must put that call into "tool_calls". Do not replace the tool with your own answer, even when you can answer from the image or from your knowledge."""

REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": ["string", "null"]},
        "tool_calls": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["text", "tool_calls"],
}

DATA_URL = re.compile(r"^data:(image/[a-z0-9.+-]+);base64,(.*)$", re.I | re.S)
# Image types the model accepts, and the name it expects for each.
MEDIA_TYPES = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "image/gif": "image/gif",
    "image/webp": "image/webp",
}


class BadRequest(Exception):
    pass


class TooLarge(BadRequest):
    """A size or count limit of the bridge is passed (HTTP 413)."""


@dataclass
class Plan:
    """What the bridge must do for one request."""

    kind: str  # "probe" | "classifier" | "tools" | "schema" | "text"
    system_prompt: str = ""
    prompt: str = ""
    json_schema: dict[str, Any] | None = None
    tool_names: set[str] = field(default_factory=set)
    # JSON schema of each caller tool, to fix argument types (see coerce_arguments).
    tool_schemas: dict[str, Any] = field(default_factory=dict)
    # Images of the request, in transcript order: {"media_type", "data"} (base64).
    images: list[dict[str, str]] = field(default_factory=list)
    json_mode: bool = False  # response_format json_object: cut the JSON out of the reply
    # The start of `prompt` that becomes its own content block (the <tools>
    # block): it is the same in every turn of a chat, so it can carry a cache mark.
    cache_prefix: str = ""

    def content_blocks(self, cache_ttl: str | None = None) -> list[dict[str, Any]]:
        """The prompt as message content: text blocks, with each image block at its placeholder.

        With a cache prefix, that prefix is the first block; `cache_ttl` ("1h"
        or "5m") puts a cache_control mark with that lifetime on it, and only on it.
        """
        if not self.cache_prefix:
            return content_blocks(self.prompt, self.images)
        head: dict[str, Any] = {"type": "text", "text": self.cache_prefix}
        if cache_ttl:
            head["cache_control"] = {"type": "ephemeral", "ttl": cache_ttl}
        return [head] + content_blocks(self.prompt[len(self.cache_prefix):], self.images)


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content)


def _placeholder(n: int) -> str:
    return f"[image {n} attached]"


class _Images:
    """Collects the images of one request as base64 blocks, in transcript order."""

    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []
        self.total = 0

    def add(self, url: str) -> str:
        m = DATA_URL.match(url.strip())
        if not m:
            return "[image: remote URL, not available to the model]"
        if len(self.items) >= cfg.max_images:
            raise TooLarge(f"more than {cfg.max_images} images in one request")
        mime = m.group(1).lower()
        encoded = m.group(2)
        # Base64 gives 3 bytes for 4 characters: refuse before the decode
        # allocates the image.
        estimate = len(encoded) * 3 // 4
        if estimate > cfg.max_image_bytes:
            raise TooLarge(f"image is larger than {cfg.max_image_bytes} bytes")
        if self.total + estimate > cfg.max_total_image_bytes:
            raise TooLarge(f"images are larger than {cfg.max_total_image_bytes} bytes in total")
        media_type = MEDIA_TYPES.get(mime)
        if media_type is None:
            raise BadRequest(f"image type {mime} is not supported")
        try:
            raw = base64.b64decode(encoded, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise BadRequest("image is not valid base64") from exc
        if not raw:
            raise BadRequest("image is empty")
        if len(raw) > cfg.max_image_bytes:
            raise TooLarge(f"image is larger than {cfg.max_image_bytes} bytes")
        self.total += len(raw)
        if self.total > cfg.max_total_image_bytes:
            raise TooLarge(f"images are larger than {cfg.max_total_image_bytes} bytes in total")
        # Re-encode: the model wants clean base64, the data URL may hold whitespace.
        self.items.append({"media_type": media_type, "data": base64.b64encode(raw).decode("ascii")})
        return _placeholder(len(self.items))


def content_blocks(prompt: str, images: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Split the prompt at the image placeholders and put each image block there."""
    blocks: list[dict[str, Any]] = []
    rest = prompt
    for n, image in enumerate(images, start=1):
        head, marker, rest = rest.partition(_placeholder(n))
        text = head + marker
        if text:
            blocks.append({"type": "text", "text": text})
        blocks.append(
            {"type": "image", "source": {"type": "base64", "media_type": image["media_type"], "data": image["data"]}}
        )
    if rest or not blocks:
        blocks.append({"type": "text", "text": rest})
    return blocks


def _render_content(content: Any, images: _Images) -> str:
    if isinstance(content, list):
        out = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                out.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                iu = part.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                if isinstance(url, str):
                    out.append(images.add(url))
        return "\n".join(x for x in out if x)
    return _text_of(content)


def _render_transcript(messages: list[dict[str, Any]], images: _Images) -> tuple[str, str]:
    system_parts: list[str] = []
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role in ("system", "developer"):
            system_parts.append(_text_of(msg.get("content")))
        elif role == "user":
            lines.append(f"[user]\n{_render_content(msg.get('content'), images)}")
        elif role == "assistant":
            text = _text_of(msg.get("content"))
            if text:
                lines.append(f"[assistant]\n{text}")
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                lines.append(
                    f"[assistant tool call id={call.get('id')}] {fn.get('name')} {fn.get('arguments') or '{}'}"
                )
        elif role == "tool":
            lines.append(
                f"[tool result for id={msg.get('tool_call_id')}]\n{_render_content(msg.get('content'), images)}"
            )
    return "\n\n".join(p for p in system_parts if p), "\n\n".join(lines)


def _tools_block(tools: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    compact = []
    schemas: dict[str, Any] = {}
    for t in tools:
        fn = t.get("function") if t.get("type") == "function" else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            continue
        schemas[fn["name"]] = fn.get("parameters", {"type": "object"})
        compact.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object"}),
            }
        )
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":")), schemas


def _tool_choice_rule(choice: Any) -> str:
    if choice == "required":
        return "You must call at least one tool in this reply."
    if isinstance(choice, dict):
        fn = choice.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name:
            return f'You must call the tool "{name}" in this reply.'
    return ""


def plan_request(body: dict[str, Any]) -> Plan:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages are required")
    if not all(isinstance(m, dict) for m in messages):
        raise BadRequest("each message must be an object")
    rf = body.get("response_format") or {}
    if not isinstance(rf, dict):
        raise BadRequest("response_format must be an object")
    tools = body.get("tools") or []
    if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
        raise BadRequest("tools must be a list of objects")
    tool_choice = body.get("tool_choice")
    schema = None
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema")
        schema = js.get("schema") if isinstance(js, dict) else None
        if not isinstance(schema, dict):
            raise BadRequest("response_format.json_schema.schema is missing")

    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    system_text = " ".join(
        _text_of(m.get("content")) for m in messages if m.get("role") in ("system", "developer")
    )
    if last_user is not None and _text_of(last_user.get("content")).strip() == PROBE_TEXT:
        return Plan(kind="probe")

    images = _Images()
    client_system, transcript = _render_transcript(messages, images)
    return _build_plan(client_system, transcript, system_text, schema, rf, tools, tool_choice, images.items)


def _build_plan(
    client_system: str,
    transcript: str,
    system_text: str,
    schema: dict[str, Any] | None,
    rf: dict[str, Any],
    tools: list[dict[str, Any]],
    tool_choice: Any,
    images: list[dict[str, str]],
) -> Plan:
    if CLASSIFIER_MARKER in system_text:
        return Plan(
            kind="classifier",
            system_prompt=client_system,
            prompt=f"<conversation>\n{transcript}\n</conversation>",
            images=images,
        )

    if schema is not None:
        return Plan(
            kind="schema",
            system_prompt=client_system or "Answer the request.",
            prompt=transcript + "\n\nReply only with data that matches the required JSON schema.",
            json_schema=schema,
            images=images,
        )

    if tools and tool_choice != "none":
        tools_json, schemas = _tools_block(tools)
        rule = _tool_choice_rule(tool_choice)
        tools_text = f"<tools>\n{tools_json}\n</tools>"
        prompt = (
            f"{tools_text}\n\n<conversation>\n{transcript}\n</conversation>\n\n"
            f"Write the next assistant reply. {rule}".strip()
        )
        return Plan(
            kind="tools",
            system_prompt=f"{client_system}\n\n{BRIDGE_RULES}".strip(),
            prompt=prompt,
            json_schema=REPLY_SCHEMA,
            tool_names=set(schemas),
            tool_schemas=schemas,
            images=images,
            cache_prefix=tools_text,
        )

    json_mode = rf.get("type") == "json_object"
    extra = "\n\nReply with one JSON object only, no other text." if json_mode else ""
    return Plan(
        kind="text",
        system_prompt=client_system or "You are a helpful assistant.",
        prompt=f"<conversation>\n{transcript}\n</conversation>\n\nWrite the next assistant reply.{extra}",
        images=images,
        json_mode=json_mode,
    )


def tool_calls_from(
    structured: Any, allowed: set[str], schemas: dict[str, Any] | None = None
) -> tuple[str | None, list[dict[str, Any]], list[tuple[str, str]]]:
    """Map the forced reply shape to (text, OpenAI tool_calls, invalid calls).

    A call whose arguments do not match the schema of its tool (after the
    repairs of coerce_arguments) is not in tool_calls: it is in the third
    list as (tool name, error) for the log and for a second try.
    """
    if not isinstance(structured, dict):
        return (str(structured) if structured is not None else None), [], []
    text = structured.get("text")
    calls = []
    invalid = []
    for call in structured.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if name not in allowed:
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        if args is None:
            args = {}
        schema = (schemas or {}).get(name)
        if schema is not None:
            args = coerce_arguments(args, schema)
            error = argument_error(args, schema)
            if error is not None:
                invalid.append((name, error))
                continue
        args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        calls.append(
            {
                "id": f"call_{secrets.token_hex(12)}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )
    return (text if isinstance(text, str) and text.strip() else None), calls, invalid


# All formats the library can check (uuid, date, ...), also for draft-07
# schemas: the caller checks them too, whatever the draft says.
_FORMATS = jsonschema.FormatChecker()


def argument_error(value: Any, schema: Any) -> str | None:
    """The first point where `value` does not match `schema`, as "path (keyword)".

    The caller (SparkyFitness) checks the arguments of a tool call against
    the same schema and refuses a call that does not match, so the bridge
    checks first. The result names only the position and the schema keyword,
    never the value, so it can go into the log. None when the value matches,
    or when the schema itself is not usable.
    """
    if not isinstance(schema, dict):
        return None
    cls = validator_for(schema, default=jsonschema.Draft7Validator)
    try:
        cls.check_schema(schema)
    except jsonschema.SchemaError:
        return None
    error = best_match(cls(schema, format_checker=_FORMATS).iter_errors(value))
    if error is None:
        return None
    path = "/".join(str(p) for p in error.absolute_path)
    return f"{path or '.'} ({error.validator})"


_NUMBER = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


def _types(schema: dict[str, Any]) -> set[str]:
    """The JSON types a schema allows, also through anyOf / oneOf."""
    out: set[str] = set()
    t = schema.get("type")
    if isinstance(t, str):
        out.add(t)
    elif isinstance(t, list):
        out.update(x for x in t if isinstance(x, str))
    for key in ("anyOf", "oneOf"):
        for sub in schema.get(key) or []:
            if isinstance(sub, dict):
                out |= _types(sub)
    return out


def _branch(schema: dict[str, Any], kind: str) -> dict[str, Any] | None:
    """The (sub)schema that describes an object or an array."""
    if kind in _types({"type": schema.get("type")}) or (kind == "object" and "properties" in schema):
        return schema
    for key in ("anyOf", "oneOf"):
        for sub in schema.get(key) or []:
            if isinstance(sub, dict) and (b := _branch(sub, kind)) is not None:
                return b
    return None


def coerce_arguments(value: Any, schema: Any) -> Any:
    """Fix values that the model wrote as strings: "27000" for a number, or
    "[\"goals\"]" for an array.

    The reply schema cannot carry the schemas of all caller tools, so claude
    does not check argument types. A native call of a caller tool is worse:
    claude has no schema for that tool at all, and the model then writes
    numbers and whole arrays as strings. A caller that validates its tool
    input (SparkyFitness uses zod) would refuse such a call. Only a string
    where the schema allows no string is changed; all other values stay as
    they are.
    """
    if not isinstance(schema, dict):
        return value
    if isinstance(value, dict):
        obj = _branch(schema, "object")
        props = obj.get("properties") if obj else None
        if isinstance(props, dict):
            return {k: coerce_arguments(v, props.get(k)) for k, v in value.items()}
        return value
    if isinstance(value, list):
        arr = _branch(schema, "array")
        items = arr.get("items") if arr else None
        return [coerce_arguments(v, items) for v in value] if isinstance(items, dict) else value
    if not isinstance(value, str):
        return value
    allowed = _types(schema)
    if not allowed or "string" in allowed:
        return value
    s = value.strip()
    if ("integer" in allowed or "number" in allowed) and _NUMBER.match(s):
        number = float(s)
        if "integer" in allowed and number.is_integer():
            return int(number)
        if "number" in allowed:
            return int(number) if number.is_integer() and "." not in s and "e" not in s.lower() else number
    if "boolean" in allowed and s.lower() in ("true", "false"):
        return s.lower() == "true"
    if "null" in allowed and s.lower() == "null":
        return None
    if ("array" in allowed or "object" in allowed) and s[:1] in ("[", "{"):
        # A whole array or object as a JSON string: parse it, then fix its items.
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = None
        if (isinstance(parsed, list) and "array" in allowed) or (isinstance(parsed, dict) and "object" in allowed):
            return coerce_arguments(parsed, schema)
    if "array" in allowed:
        return _string_as_array(value, schema)
    return value


def _string_as_array(value: str, schema: dict[str, Any]) -> Any:
    """One string where the schema wants an array: ["x"] when "x" is a valid
    item, ["x", "y"] for "x, y" when the items are an enum and each part is in it."""
    arr = _branch(schema, "array")
    items = arr.get("items") if arr else None
    if not isinstance(items, dict):
        return value
    if argument_error(value, items) is None:
        return [value]
    parts = [p.strip() for p in value.split(",")]
    if "enum" in items and len(parts) > 1 and all(argument_error(p, items) is None for p in parts):
        return parts
    return value


def json_text(result_text: str) -> str:
    """For json_object requests: cut the JSON object out of the reply text."""
    s = result_text.strip()
    # Parse from the first "{" that starts a full object, so fences or braces
    # inside JSON strings cannot cut the object short.
    decoder = json.JSONDecoder()
    start = s.find("{")
    while start != -1:
        try:
            obj, end = decoder.raw_decode(s, start)
        except json.JSONDecodeError:
            start = s.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return s[start:end]
        start = s.find("{", end)
    return s
