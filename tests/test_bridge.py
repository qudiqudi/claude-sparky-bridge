import base64
import json
import os
import time

from tests.conftest import AUTH, BASE, assistant_event, envelope, tool_use

URL = "/v1/chat/completions"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sparky_log_food",
            "description": "Log a food",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
    }
]
JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0fakejpeg").decode()
JPEG = "data:image/jpeg;base64," + JPEG_B64
PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakepng").decode()
PNG = "data:image/png;base64," + PNG_B64


def _blocks(seen):
    return seen()["message"]["message"]["content"]


def _prompt(seen):
    """The text that went to claude: all text blocks of the stdin message."""
    return "".join(b["text"] for b in _blocks(seen) if b["type"] == "text")


def _disk_files():
    """Every file under the test base, except the files of the fake and the counter."""
    out = set()
    for root, _, files in os.walk(BASE):
        for f in files:
            out.add(os.path.join(root, f))
    return {f for f in out if os.path.basename(f) not in ("reply.json", "seen.json", "runs.log", "claude", "counter.json", "counter.json.tmp")}


def _sse(text):
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            events.append(block[6:])
    return events


def test_auth_required(client, reply):
    r = client.post(URL, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401
    r = client.post(URL, json={"messages": [{"role": "user", "content": "hi"}]}, headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_probe_answers_without_claude(client, reply, seen):
    reply(envelope("should not be used"))
    r = client.post(URL, headers=AUTH, json={"model": "sonnet", "messages": [{"role": "user", "content": "Reply with the single word: OK."}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "OK"
    assert seen() is None


def test_plain_text(client, reply, seen):
    reply(envelope("Hallo!"))
    r = client.post(URL, headers=AUTH, json={"model": "sonnet", "messages": [{"role": "system", "content": "Be nice"}, {"role": "user", "content": "hi"}]})
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hallo!"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127}
    s = seen()
    argv = s["argv"]
    assert argv[argv.index("--system-prompt") + 1] == "Be nice"
    assert argv[argv.index("--tools") + 1] == ""
    assert "--json-schema" not in argv
    assert "--strict-mcp-config" in argv and "--no-session-persistence" in argv and "--verbose" in argv
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert argv[argv.index("--settings") + 1].endswith("settings.json")
    # One user message line on stdin, nothing after it.
    assert s["message"]["type"] == "user" and s["message"]["message"]["role"] == "user"
    assert _blocks(seen) == [{"type": "text", "text": _prompt(seen)}]
    assert s["stdin_rest"] == ""


def test_tool_call_mapping_filters_unknown_tools(client, reply, seen):
    reply(envelope(structured={"text": None, "tool_calls": [
        {"name": "sparky_log_food", "arguments": {"name": "Skyr"}},
        {"name": "not_a_tool", "arguments": {}},
    ]}))
    r = client.post(URL, headers=AUTH, json={"model": "sonnet", "tools": TOOLS, "messages": [{"role": "user", "content": "Ich hatte Skyr"}]})
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "sparky_log_food"
    assert json.loads(calls[0]["function"]["arguments"]) == {"name": "Skyr"}
    assert calls[0]["id"].startswith("call_")
    prompt = _prompt(seen)
    assert "sparky_log_food" in prompt and "<tools>" in prompt
    assert "--json-schema" in seen()["argv"]


def test_transcript_has_tool_results(client, reply, seen):
    reply(envelope(structured={"text": "Eingetragen.", "tool_calls": None}))
    messages = [
        {"role": "user", "content": "Ich hatte Skyr"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "sparky_log_food", "arguments": "{\"name\":\"Skyr\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\":true}"},
    ]
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": messages})
    assert r.json()["choices"][0]["message"]["content"] == "Eingetragen."
    prompt = _prompt(seen)
    assert "[assistant tool call id=call_1] sparky_log_food" in prompt
    assert "[tool result for id=call_1]" in prompt


def test_stream_tool_call(client, reply):
    reply(envelope(structured={"text": None, "tool_calls": [{"name": "sparky_log_food", "arguments": {"name": "Ei"}}]}))
    with client.stream("POST", URL, headers=AUTH, json={"stream": True, "stream_options": {"include_usage": True}, "tools": TOOLS, "messages": [{"role": "user", "content": "2 Eier"}]}) as r:
        text = "".join(r.iter_text())
    events = _sse(text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    tc = [c for c in chunks if c["choices"] and "tool_calls" in c["choices"][0]["delta"]]
    assert tc[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert tc[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "sparky_log_food"
    finish = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finish[0]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 7


def test_stream_text(client, reply):
    reply(envelope(structured={"text": "Hallo", "tool_calls": None}))
    with client.stream("POST", URL, headers=AUTH, json={"stream": True, "tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]}) as r:
        events = _sse("".join(r.iter_text()))
    chunks = [json.loads(e) for e in events[:-1]]
    contents = [c["choices"][0]["delta"].get("content") for c in chunks if c["choices"]]
    assert "Hallo" in contents
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_json_schema_vision_with_image(client, reply, seen):
    reply(envelope(structured={"items": [{"name": "Banane", "kcal": 90}]}))
    schema = {"type": "object", "properties": {"items": {"type": "array"}}, "required": ["items"]}
    body = {
        "model": "sonnet",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Estimate"}, {"type": "image_url", "image_url": {"url": JPEG}}]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": schema}},
    }
    r = client.post(URL, headers=AUTH, json=body)
    content = r.json()["choices"][0]["message"]["content"]
    assert json.loads(content) == {"items": [{"name": "Banane", "kcal": 90}]}
    argv = seen()["argv"]
    assert argv[argv.index("--tools") + 1] == ""
    assert json.loads(argv[argv.index("--json-schema") + 1]) == schema
    blocks = _blocks(seen)
    assert [b["type"] for b in blocks] == ["text", "image", "text"]
    assert blocks[0]["text"].endswith("Estimate\n[image 1 attached]")
    assert blocks[1]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": JPEG_B64}
    assert blocks[2]["text"].startswith("\n")
    # The photo never touched the disk.
    assert _disk_files() == set()


def _image_body(n=1):
    parts = [{"type": "text", "text": "Estimate"}] + [{"type": "image_url", "image_url": {"url": JPEG}}] * n
    return {"messages": [{"role": "user", "content": parts}]}


def test_json_object_text_is_cut(client, reply):
    reply(envelope("Hier: ```json\n{\"a\": 1}\n```"))
    r = client.post(URL, headers=AUTH, json={"response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": "x"}]})
    assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"a": 1}


def test_plain_text_with_code_fence_is_kept(client, reply):
    text = "```json\n{\"a\": 1}\n```\nDas ist das Format."
    reply(envelope(text))
    r = client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "x"}]})
    assert r.json()["choices"][0]["message"]["content"] == text


def test_classifier_uses_fast_model(client, reply, seen):
    reply(envelope("food"))
    system = "Analyze the conversation history (especially the user's latest reply) and determine which of the following health tracking domains are relevant."
    r = client.post(URL, headers=AUTH, json={"model": "sonnet", "messages": [{"role": "system", "content": system}, {"role": "user", "content": "Was hatte ich gestern?"}]})
    assert r.json()["choices"][0]["message"]["content"] == "food"
    argv = seen()["argv"]
    assert argv[argv.index("--model") + 1] == "haiku"


def test_daily_limit(client, reply):
    reply(envelope("ok"))
    (BASE / "counter.json").write_text(json.dumps({"date": __import__("datetime").date.today().isoformat(), "count": 50}))
    r = client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 429


def test_claude_error_is_reported(client, reply):
    reply(envelope("rate limited", is_error=True))
    r = client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert "rate limited" in r.json()["error"]["message"]


def test_unknown_model_falls_back(client, reply, seen):
    reply(envelope("ok"))
    client.post(URL, headers=AUTH, json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    argv = seen()["argv"]
    assert argv[argv.index("--model") + 1] == "sonnet"


def test_healthz(client):
    assert client.get("/healthz").json()["ok"] is True


def test_rejected_image_request_gives_400(client, reply, seen):
    body = {
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": JPEG}}]}],
        "response_format": {"type": "json_schema", "json_schema": {}},
    }
    r = client.post(URL, headers=AUTH, json=body)
    assert r.status_code == 400
    assert seen() is None
    assert _disk_files() == set()


def test_image_block_order_with_two_images(client, reply, seen):
    reply(envelope(structured={"text": "zwei Bilder", "tool_calls": None}))
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Erstes:"}, {"type": "image_url", "image_url": {"url": JPEG}}, {"type": "text", "text": "Zweites:"}, {"type": "image_url", "image_url": {"url": PNG}}, {"type": "text", "text": "Vergleiche."}]},
    ]
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": messages})
    assert r.status_code == 200
    blocks = _blocks(seen)
    # The <tools> block comes first, on its own; the transcript follows with the images in place.
    assert [b["type"] for b in blocks] == ["text", "text", "image", "text", "image", "text"]
    assert blocks[0]["text"].startswith("<tools>\n") and blocks[0]["text"].endswith("\n</tools>")
    assert blocks[1]["text"].startswith("\n\n<conversation>") and blocks[1]["text"].endswith("Erstes:\n[image 1 attached]")
    assert blocks[2]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": JPEG_B64}
    assert blocks[3]["text"] == "\nZweites:\n[image 2 attached]"
    assert blocks[4]["source"] == {"type": "base64", "media_type": "image/png", "data": PNG_B64}
    assert blocks[5]["text"].startswith("\nVergleiche.")
    assert "Write the next assistant reply." in blocks[5]["text"]
    assert _disk_files() == set()


def test_remote_image_url_is_a_placeholder(client, reply, seen):
    reply(envelope("ok"))
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/a.jpg"}}]}]}
    assert client.post(URL, headers=AUTH, json=body).status_code == 200
    assert [b["type"] for b in _blocks(seen)] == ["text"]
    assert "[image: remote URL, not available to the model]" in _prompt(seen)


def test_tools_system_prompt_has_image_rules(client, reply, seen):
    reply(envelope(structured={"text": None, "tool_calls": [{"name": "sparky_log_food", "arguments": {"name": "x"}}]}))
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": [{"type": "text", "text": "Foto"}, {"type": "image_url", "image_url": {"url": JPEG}}]}]})
    assert r.status_code == 200
    argv = seen()["argv"]
    system = argv[argv.index("--system-prompt") + 1]
    assert "Images are attached to the conversation." in system
    assert 'you must put that call into "tool_calls"' in system
    assert "Read tool" not in system


# --- native tool calls in the stream -----------------------------------------


def _native_events(name, arguments, text_blocks=(), usage=None, sleep=5):
    """claude calls a caller tool natively, then would hang around for `sleep` seconds."""
    blocks = [{"type": "text", "text": t} for t in text_blocks] + [tool_use(name, arguments)]
    return [
        {"type": "system", "subtype": "init", "tools": []},
        assistant_event(blocks, usage),
        {"_sleep": sleep},
        envelope(structured={"text": "too late", "tool_calls": None}),
    ]


def test_native_tool_use_is_returned_and_claude_is_killed(client, reply):
    reply(_native_events("sparky_log_food", {"name": "Skyr"}, ("Ich trage das ein.",), usage={"input_tokens": 50, "output_tokens": 9}))
    t0 = time.monotonic()
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "Ich hatte Skyr"}]})
    elapsed = time.monotonic() - t0
    assert elapsed < 2, elapsed
    body = r.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Ich trage das ein."
    calls = choice["message"]["tool_calls"]
    assert len(calls) == 1 and calls[0]["id"].startswith("call_")
    assert calls[0]["function"]["name"] == "sparky_log_food"
    assert json.loads(calls[0]["function"]["arguments"]) == {"name": "Skyr"}
    assert body["usage"] == {"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59}


def test_native_tool_use_without_usage_gives_zeros(client, reply):
    reply(_native_events("sparky_log_food", {"name": "Ei"}))
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "Ei"}]})
    body = r.json()
    assert body["choices"][0]["message"]["content"] is None
    assert body["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_native_tool_use_streams_tool_calls(client, reply):
    reply(_native_events("sparky_log_food", {"name": "Ei"}))
    t0 = time.monotonic()
    with client.stream("POST", URL, headers=AUTH, json={"stream": True, "tools": TOOLS, "messages": [{"role": "user", "content": "2 Eier"}]}) as r:
        events = _sse("".join(r.iter_text()))
    # The fake sleeps 5 s after the result; less than that proves no wait for it.
    assert time.monotonic() - t0 < 4
    chunks = [json.loads(e) for e in events[:-1]]
    tc = [c for c in chunks if c["choices"] and "tool_calls" in c["choices"][0]["delta"]]
    assert tc[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "sparky_log_food"
    assert events[-1] == "[DONE]"


def test_unknown_native_tool_use_is_ignored(client, reply):
    reply([
        assistant_event([tool_use("Read", {"file_path": "/etc/passwd"})]),
        {"type": "user", "message": {"role": "user", "content": []}},
        assistant_event([tool_use("StructuredOutput", {"text": "Fertig.", "tool_calls": None})]),
        envelope(structured={"text": "Fertig.", "tool_calls": None}),
    ])
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "x"}]})
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Fertig."
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 120


def test_structured_output_path_still_works(client, reply):
    reply([
        {"type": "system", "subtype": "init"},
        assistant_event([tool_use("StructuredOutput", {"text": None, "tool_calls": [{"name": "sparky_log_food", "arguments": {"name": "Skyr"}}]})]),
        envelope(structured={"text": None, "tool_calls": [{"name": "sparky_log_food", "arguments": {"name": "Skyr"}}]}),
    ])
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "Skyr"}]})
    calls = r.json()["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == {"name": "Skyr"}


def test_native_tool_use_is_not_intercepted_without_tools(client, reply):
    # A text request has no caller tools: the stream runs to the result event.
    reply([
        assistant_event([tool_use("sparky_log_food", {"name": "x"})]),
        envelope("Hallo"),
    ])
    r = client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["choices"][0]["message"]["content"] == "Hallo"


def test_result_event_missing_is_an_error(client, reply):
    reply([{"type": "system", "subtype": "init"}, assistant_event([{"type": "text", "text": "x"}])])
    r = client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert "no result event" in r.json()["error"]["message"]


def test_result_event_missing_reports_the_exit_code(client, reply):
    reply([{"type": "system", "subtype": "init"}, {"_exit": 3}])
    r = client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert "exit code 3" in r.json()["error"]["message"]


# --- the run ends at the result event ----------------------------------------


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_result_event_returns_at_once_and_claude_is_killed(client, reply, seen):
    # claude would hang around for 5 s after its result; the bridge does not wait.
    reply([envelope("Hallo"), {"_sleep": 5}, {"_raw": "late"}])
    t0 = time.monotonic()
    r = client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    # The fake sleeps 5 s after the result; less than that proves no wait for it.
    assert time.monotonic() - t0 < 4
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Hallo"
    assert not _alive(seen()["pid"])


def test_exit_code_after_the_result_event_is_not_an_error(client, reply):
    reply([envelope(structured={"text": "ok", "tool_calls": None}), {"_exit": 7}])
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok"


def test_error_in_the_result_event_returns_at_once(client, reply):
    reply([envelope("kaputt", is_error=True), {"_sleep": 5}])
    t0 = time.monotonic()
    r = client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    # The fake sleeps 5 s after the result; less than that proves no wait for it.
    assert time.monotonic() - t0 < 4
    assert r.status_code == 502
    assert "kaputt" in r.json()["error"]["message"]


def _log_lines(caplog):
    return [rec.getMessage() for rec in caplog.records if rec.name == "bridge" and rec.levelname == "INFO"]


def test_log_line_has_cache_and_api_values(client, reply, caplog):
    import logging

    env = envelope("Hallo")
    env["usage"] = {"input_tokens": 5, "cache_creation_input_tokens": 300, "cache_read_input_tokens": 4000, "output_tokens": 9}
    env["duration_api_ms"] = 1234
    reply(env)
    with caplog.at_level(logging.INFO, logger="bridge"):
        r = client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["usage"]["prompt_tokens"] == 4305
    line = next(l for l in _log_lines(caplog) if l.startswith("kind=text"))
    for part in ("duration_api_ms=1234", "tokens_in=4305", "input_tokens=5", "cache_creation_input_tokens=300",
                 "cache_read_input_tokens=4000", "tokens_out=9"):
        assert part in line, line
    assert "Hallo" not in line


def test_log_line_on_the_intercept_path_has_no_api_time(client, reply, caplog):
    import logging

    usage = {"input_tokens": 3, "cache_creation_input_tokens": 40, "cache_read_input_tokens": 500, "output_tokens": 8}
    reply(_native_events("sparky_log_food", {"name": "Ei"}, usage=usage))
    with caplog.at_level(logging.INFO, logger="bridge"):
        r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "Ei"}]})
    assert r.status_code == 200
    line = next(l for l in _log_lines(caplog) if l.startswith("kind=tools"))
    for part in ("duration_api_ms=-", "turns=intercepted", "tokens_in=543", "input_tokens=3",
                 "cache_creation_input_tokens=40", "cache_read_input_tokens=500", "tokens_out=8"):
        assert part in line, line


# --- cache mark on the tools block --------------------------------------------


CACHE_400 = "API Error: 400 A maximum of 4 blocks with cache_control may be provided. Found 5."


def _marked(blocks):
    return [b for b in blocks if "cache_control" in b]


def test_tools_block_is_its_own_block_with_one_cache_mark(client, reply, seen):
    reply(envelope(structured={"text": "ok", "tool_calls": None}))
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    blocks = _blocks(seen)
    assert [b["type"] for b in blocks] == ["text", "text"]
    assert blocks[0]["text"].startswith("<tools>\n") and blocks[0]["text"].endswith("\n</tools>")
    assert "sparky_log_food" in blocks[0]["text"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert blocks[1]["text"].startswith("\n\n<conversation>\n[user]\nhi")
    assert _marked(blocks) == [blocks[0]]


def test_cache_mark_only_on_tools_requests(client, reply, seen):
    reply(envelope("Hallo"))
    client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    assert _marked(_blocks(seen)) == []
    reply(envelope(structured={"a": 1}))
    body = {"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}}
    client.post(URL, headers=AUTH, json=body)
    assert _marked(_blocks(seen)) == []


def test_cache_mark_off_by_setting(client, reply, seen):
    reply(envelope(structured={"text": "ok", "tool_calls": None}))
    with _with_cfg(cache_tools=False):
        r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    blocks = _blocks(seen)
    assert _marked(blocks) == []
    assert "<tools>" in blocks[0]["text"] and blocks[1]["text"].startswith("\n\n<conversation>")


def test_cache_tools_setting_parses_off_values(monkeypatch):
    from app.config import _bool

    for raw, want in (("", True), ("1", True), ("true", True), ("0", False), ("false", False), ("False", False)):
        monkeypatch.setenv("BRIDGE_CACHE_TOOLS", raw)
        assert _bool("BRIDGE_CACHE_TOOLS", True) is want, raw


TTL_400 = "API Error: 400 messages.1.content.0.cache_control.ttl: a ttl='1h' cache_control block must not come after a ttl='5m' cache_control block."


def _mark_ttls(seen):
    return [b["cache_control"].get("ttl") for b in _marked(_blocks(seen))]


def test_refused_1h_mark_tries_5m_and_keeps_it(client, reply, seen, runs, caplog):
    import logging

    reply({"_queue": [envelope(TTL_400, is_error=True), envelope(structured={"text": "ok", "tool_calls": None})]})
    with caplog.at_level(logging.WARNING, logger="bridge"):
        r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok"
    assert runs() == 2
    assert _mark_ttls(seen) == ["5m"]
    warnings = [rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"]
    assert len(warnings) == 1 and "cache mark" in warnings[0]
    # The next chat starts with the choice that worked: one run, 5m mark.
    reply(envelope(structured={"text": "again", "tool_calls": None}))
    client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]})
    assert runs() == 3
    assert _mark_ttls(seen) == ["5m"]


def test_refused_marks_fall_back_to_no_mark_and_keep_it(client, reply, seen, runs):
    reply({"_queue": [envelope(CACHE_400, is_error=True), envelope(CACHE_400, is_error=True), envelope(structured={"text": "ok", "tool_calls": None})]})
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert runs() == 3
    blocks = _blocks(seen)
    assert _marked(blocks) == [] and "<tools>" in blocks[0]["text"]
    reply(envelope(structured={"text": "again", "tool_calls": None}))
    client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]})
    assert runs() == 4
    assert _marked(_blocks(seen)) == []


def test_refused_cache_mark_does_not_loop(client, reply, runs):
    reply({"_queue": [envelope(CACHE_400, is_error=True)] * 3 + [envelope("never")]})
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert "cache_control" in r.json()["error"]["message"]
    assert runs() == 3


def test_other_result_errors_do_not_run_again(client, reply, runs):
    reply({"_queue": [envelope("rate limited", is_error=True), envelope(structured={"text": "ok", "tool_calls": None})]})
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert runs() == 1


def test_refused_cache_mark_without_mark_is_not_run_again(client, reply, runs):
    with _with_cfg(cache_tools=False):
        reply({"_queue": [envelope(CACHE_400, is_error=True), envelope(structured={"text": "ok", "tool_calls": None})]})
        r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert runs() == 1


def test_non_json_lines_are_ignored(client, reply):
    reply([{"_raw": "not json at all"}, {"_raw": "[1, 2]"}, envelope("Hallo")])
    r = client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["choices"][0]["message"]["content"] == "Hallo"


def test_json_object_keeps_fences_inside_strings(client, reply):
    obj = {"example": "```json {} ```", "answer": 42}
    raw = json.dumps(obj)
    for text in (raw, f"```json\n{raw}\n```", f"Hier ist es: {raw}", f"Hier:\n```json\n{raw}\n```\nFertig."):
        reply(envelope(text))
        r = client.post(URL, headers=AUTH, json={"response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": "x"}]})
        assert json.loads(r.json()["choices"][0]["message"]["content"]) == obj, text


def test_malformed_bodies_give_400(client, reply):
    reply(envelope("ok"))
    for body in (
        {"messages": [None]},
        {"messages": [{"role": "user", "content": "hi"}], "response_format": "json"},
        {"messages": [{"role": "user", "content": "hi"}], "tools": "x"},
        {"messages": [{"role": "assistant", "content": None, "tool_calls": ["x"]}, {"role": "user", "content": "hi"}]},
    ):
        r = client.post(URL, headers=AUTH, json=body)
        assert r.status_code == 400, body


def test_non_string_model_falls_back(client, reply, seen):
    reply(envelope("ok"))
    r = client.post(URL, headers=AUTH, json={"model": 123, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    argv = seen()["argv"]
    assert argv[argv.index("--model") + 1] == "sonnet"


def test_stream_options_not_a_dict(client, reply):
    reply(envelope("Hallo"))
    with client.stream("POST", URL, headers=AUTH, json={"stream": True, "stream_options": [1], "messages": [{"role": "user", "content": "hi"}]}) as r:
        events = _sse("".join(r.iter_text()))
    assert events[-1] == "[DONE]"


def test_tools_reply_without_text_or_known_call_is_an_error(client, reply):
    reply(envelope(structured={"text": None, "tool_calls": [{"name": "not_a_tool", "arguments": {}}]}))
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 502


def test_timeout_kills_claude(client, reply):
    from app.config import cfg
    object.__setattr__(cfg, "timeout_s", 1)
    try:
        env = envelope("late")
        env["_sleep"] = 5
        reply(env)
        r = client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 504
    finally:
        object.__setattr__(cfg, "timeout_s", 240)


def test_stream_error_has_error_event_and_done(client, reply):
    reply(envelope("kaputt", is_error=True))
    with client.stream("POST", URL, headers=AUTH, json={"stream": True, "messages": [{"role": "user", "content": "hi"}]}) as r:
        events = _sse("".join(r.iter_text()))
    assert events[-1] == "[DONE]"
    assert json.loads(events[-2])["error"]["message"].startswith("claude reported an error")


def test_env_is_an_allow_list(client, reply, seen):
    reply(envelope("ok"))
    client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    s = seen()
    assert set(s["env"]) - {"PWD", "SHLVL", "_", "LC_CTYPE", "__CF_USER_TEXT_ENCODING"} <= {
        "CLAUDE_CODE_OAUTH_TOKEN", "HOME", "PATH", "DISABLE_AUTOUPDATER", "TZ"
    }
    assert "test-bridge" not in _prompt(seen) and "test-oauth" not in " ".join(s["argv"])


def test_huge_system_prompt_gives_413(client, reply):
    reply(envelope("ok"))
    r = client.post(URL, headers=AUTH, json={"messages": [{"role": "system", "content": "x" * 200_000}, {"role": "user", "content": "hi"}]})
    assert r.status_code == 413


# --- resource limits ---------------------------------------------------------


def _with_cfg(**values):
    from app.config import cfg
    return _CfgPatch(cfg, values)


class _CfgPatch:
    def __init__(self, cfg, values):
        self.cfg, self.values, self.old = cfg, values, {}

    def __enter__(self):
        for k, v in self.values.items():
            self.old[k] = getattr(self.cfg, k)
            object.__setattr__(self.cfg, k, v)

    def __exit__(self, *exc):
        for k, v in self.old.items():
            object.__setattr__(self.cfg, k, v)


def test_body_too_large_by_content_length(client, reply, seen):
    reply(envelope("ok"))
    body = {"messages": [{"role": "user", "content": "x" * 5000}]}
    with _with_cfg(max_body_bytes=1000):
        r = client.post(URL, headers=AUTH, json=body)
    assert r.status_code == 413
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert seen() is None


def test_body_too_large_while_streaming(client, reply, seen):
    reply(envelope("ok"))
    # A chunked body has no Content-Length; the bridge counts the chunks.
    raw = json.dumps({"messages": [{"role": "user", "content": "x" * 5000}]}).encode()
    chunks = [raw[i:i + 500] for i in range(0, len(raw), 500)]
    with _with_cfg(max_body_bytes=1000):
        r = client.post(URL, headers={**AUTH, "Content-Type": "application/json"}, content=iter(chunks))
    assert r.status_code == 413
    assert seen() is None


def test_body_within_limit_is_accepted(client, reply):
    reply(envelope("ok"))
    with _with_cfg(max_body_bytes=1000):
        r = client.post(URL, headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


def test_too_many_images(client, reply, seen):
    reply(envelope("ok"))
    with _with_cfg(max_images=1):
        r = client.post(URL, headers=AUTH, json=_image_body(2))
    assert r.status_code == 413
    assert "images" in r.json()["error"]["message"]
    assert seen() is None


def test_images_too_large_in_total(client, reply, seen):
    reply(envelope("ok"))
    # Each fake JPEG is 12 bytes: one fits, two do not.
    with _with_cfg(max_total_image_bytes=20):
        r = client.post(URL, headers=AUTH, json=_image_body(2))
        assert r.status_code == 413
        assert seen() is None
        r = client.post(URL, headers=AUTH, json=_image_body(1))
        assert r.status_code == 200


def test_image_too_large_is_refused_before_decode(client, reply, seen):
    reply(envelope("ok"))
    with _with_cfg(max_image_bytes=4):
        r = client.post(URL, headers=AUTH, json=_image_body())
    assert r.status_code == 413
    assert seen() is None


def test_image_limit_counts_decoded_bytes(client, reply, seen):
    reply(envelope("ok"))
    # The fake JPEG is 12 bytes: a limit of 11 refuses it, 12 accepts it.
    with _with_cfg(max_image_bytes=11):
        assert client.post(URL, headers=AUTH, json=_image_body()).status_code == 413
        assert seen() is None
    with _with_cfg(max_image_bytes=12):
        assert client.post(URL, headers=AUTH, json=_image_body()).status_code == 200


def test_bad_images_give_400(client, reply, seen):
    reply(envelope("ok"))
    for url in ("data:image/jpeg;base64,", "data:image/jpeg;base64,!!!!", "data:image/tiff;base64," + JPEG_B64):
        body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]}
        r = client.post(URL, headers=AUTH, json=body)
        assert r.status_code == 400, url
        assert seen() is None


def test_queue_full_gives_429(reply):
    import asyncio
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import app.main as m
    from fastapi.testclient import TestClient

    env = envelope("slow")
    env["_sleep"] = 2
    reply(env)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    old_slots = m._slots
    m._slots = asyncio.Semaphore(1)
    try:
        # One client context: all requests share one event loop, like uvicorn.
        with _with_cfg(max_queued=1), TestClient(m.app) as c, ThreadPoolExecutor(3) as pool:
            futures = []
            for _ in range(3):
                futures.append(pool.submit(c.post, URL, headers=AUTH, json=body))
                time.sleep(0.5)  # first runs, second waits, third finds the queue full
            codes = [f.result().status_code for f in futures]
    finally:
        m._slots = old_slots
    assert codes == [200, 200, 429]
    assert m.queue_state() == {"running": 0, "waiting": 0}


def test_cancelled_waiter_leaves_the_queue():
    import asyncio

    import app.main as m

    async def scenario():
        m._slots = asyncio.Semaphore(1)
        async with m._claude_slot():
            async def wait_for_slot():
                async with m._claude_slot():
                    pass
            waiter = asyncio.create_task(wait_for_slot())
            await asyncio.sleep(0)
            assert m.queue_state() == {"running": 1, "waiting": 1}
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            assert m.queue_state() == {"running": 1, "waiting": 0}
        assert m.queue_state() == {"running": 0, "waiting": 0}
        # The slot the waiter gave up is free again.
        async with m._claude_slot():
            pass

    old_slots = m._slots
    try:
        asyncio.run(scenario())
    finally:
        m._slots = old_slots


SLEEP_TOOL = {
    "type": "function",
    "function": {
        "name": "sparky_manage_checkin",
        "description": "Check-in actions",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["log_sleep"]},
                "duration_seconds": {"type": "number", "minimum": 0},
                "steps": {"type": "integer"},
                "is_nap": {"type": "boolean"},
                "note": {"type": "string"},
                "stages": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"minutes": {"type": "integer"}}},
                },
            },
        },
    },
}


def test_structured_tool_arguments_get_schema_types(client, reply):
    reply(envelope(structured={"text": None, "tool_calls": [{"name": "sparky_manage_checkin", "arguments": {
        "action": "log_sleep", "duration_seconds": "27000", "steps": "8500", "is_nap": "false",
        "note": "123", "stages": [{"minutes": "90"}],
    }}]}))
    r = client.post(URL, headers=AUTH, json={"tools": [SLEEP_TOOL], "messages": [{"role": "user", "content": "I slept 7.5 h"}]})
    args = json.loads(r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"action": "log_sleep", "duration_seconds": 27000, "steps": 8500, "is_nap": False,
                    "note": "123", "stages": [{"minutes": 90}]}


def test_native_tool_arguments_get_schema_types(client, reply):
    reply(_native_events("sparky_manage_checkin", {"action": "log_sleep", "duration_seconds": "27000.5"}, ()))
    r = client.post(URL, headers=AUTH, json={"tools": [SLEEP_TOOL], "messages": [{"role": "user", "content": "I slept"}]})
    args = json.loads(r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"action": "log_sleep", "duration_seconds": 27000.5}


def test_coerce_arguments_keeps_values_it_can_not_fix():
    from app.translate import coerce_arguments

    schema = SLEEP_TOOL["function"]["parameters"]
    value = {"duration_seconds": "about 7 hours", "is_nap": "maybe", "unknown": "5", "steps": "12.5"}
    assert coerce_arguments(value, schema) == value


def test_coerce_arguments_follows_any_of():
    from app.translate import coerce_arguments

    schema = {"type": "object", "properties": {
        "weight": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "label": {"anyOf": [{"type": "number"}, {"type": "string"}]},
        "sets": {"anyOf": [{"type": "array", "items": {"type": "object", "properties": {"reps": {"type": "integer"}}}}, {"type": "null"}]},
    }}
    out = coerce_arguments({"weight": "81.5", "label": "42", "sets": [{"reps": "10"}]}, schema)
    assert out == {"weight": 81.5, "label": "42", "sets": [{"reps": 10}]}
    assert coerce_arguments({"weight": "null"}, schema) == {"weight": None}


def test_cache_control_text_without_api_400_is_not_run_again(client, reply, runs):
    reply(envelope("claude could not read cache_control settings", is_error=True))
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert runs() == 1


# --- tool arguments that do not match the schema of the tool --------------------

ENABLE_TOOL = {
    "type": "function",
    "function": {
        "name": "sparky_enable_tools",
        "description": "Enables tool categories",
        "parameters": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "enum": ["food", "goals", "reports", "checkin"]},
                },
            },
            "required": ["categories"],
            "additionalProperties": False,
        },
    },
}


def _warnings(caplog):
    return [rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"]


def _calls(r):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in r.json()["choices"][0]["message"]["tool_calls"]]


def test_native_array_as_json_string_is_repaired(client, reply, runs):
    # Seen in production: claude has no schema for a caller tool, and the
    # model writes the array as a string. The call is taken, with the array.
    reply(_native_events("sparky_enable_tools", {"categories": '["goals", "reports"]'}))
    t0 = time.monotonic()
    r = client.post(URL, headers=AUTH, json={"tools": [ENABLE_TOOL], "messages": [{"role": "user", "content": "calories left?"}]})
    assert time.monotonic() - t0 < 2
    assert _calls(r) == [("sparky_enable_tools", {"categories": ["goals", "reports"]})]
    assert runs() == 1


def test_native_invalid_call_lets_claude_continue(client, reply, runs, caplog):
    import logging

    # The call can not be repaired: claude is not stopped, gets "no such
    # tool" from its own loop, and writes the call into the structured reply.
    good = {"text": None, "tool_calls": [{"name": "sparky_enable_tools", "arguments": {"categories": ["goals"]}}]}
    reply([
        assistant_event([tool_use("sparky_enable_tools", {"categories": "nutrition"})], usage={"input_tokens": 1, "output_tokens": 3}),
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "No such tool", "is_error": True}]}},
        assistant_event([tool_use("StructuredOutput", good)]),
        envelope(structured=good),
    ])
    with caplog.at_level(logging.INFO):
        r = client.post(URL, headers=AUTH, json={"tools": [ENABLE_TOOL], "messages": [{"role": "user", "content": "calories left?"}]})
    assert _calls(r) == [("sparky_enable_tools", {"categories": ["goals"]})]
    assert r.json()["usage"]["prompt_tokens"] == 120  # from the result event, not the native call
    assert runs() == 1
    line = next(l for l in _warnings(caplog) if l.startswith("native tool call does not match"))
    assert "sparky_enable_tools at categories (type)" in line and "claude continues" in line
    assert "nutrition" not in line


def test_native_message_with_one_invalid_call_takes_none(client, reply):
    good = {"text": None, "tool_calls": [{"name": "sparky_log_food", "arguments": {"name": "Ei"}}]}
    reply([
        assistant_event([tool_use("sparky_log_food", {"name": "Ei"}), tool_use("sparky_enable_tools", {"categories": []})]),
        envelope(structured=good),
    ])
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS + [ENABLE_TOOL], "messages": [{"role": "user", "content": "Ei"}]})
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert r.json()["usage"]["prompt_tokens"] == 120


def test_structured_invalid_call_runs_once_more(client, reply, seen, runs, caplog):
    import logging

    bad = {"text": "Let me check.", "tool_calls": [{"name": "sparky_enable_tools", "arguments": {"categories": ["nutrition"]}}]}
    good = {"text": None, "tool_calls": [{"name": "sparky_enable_tools", "arguments": {"categories": ["food", "reports"]}}]}
    reply({"_queue": [envelope(structured=bad), envelope(structured=good)]})
    with caplog.at_level(logging.INFO):
        r = client.post(URL, headers=AUTH, json={"tools": [ENABLE_TOOL], "messages": [{"role": "user", "content": "calories left?"}]})
    assert r.status_code == 200
    assert _calls(r) == [("sparky_enable_tools", {"categories": ["food", "reports"]})]
    assert r.json()["choices"][0]["message"]["content"] is None
    assert runs() == 2
    # The second run sees the refused call and the error; the log sees only the position.
    prompt = _prompt(seen)
    assert "was not accepted" in prompt and "categories/0 (enum)" in prompt and '"nutrition"' in prompt
    assert prompt.startswith("<tools>")
    line = next(l for l in _warnings(caplog) if l.startswith("tool call does not match"))
    assert line == "tool call does not match its schema, claude runs once more: sparky_enable_tools at categories/0 (enum)"
    assert sum(l.startswith("kind=tools") for l in _log_lines(caplog)) == 2


def test_structured_invalid_call_twice_is_dropped(client, reply, runs, caplog):
    import logging

    bad = {"text": "Let me check.", "tool_calls": [{"name": "sparky_enable_tools", "arguments": {"categories": "goals,bogus"}}]}
    reply({"_queue": [envelope(structured=bad), envelope(structured=bad)]})
    with caplog.at_level(logging.INFO):
        r = client.post(URL, headers=AUTH, json={"tools": [ENABLE_TOOL], "messages": [{"role": "user", "content": "calories left?"}]})
    assert r.status_code == 502
    assert "does not match the schema" in r.json()["error"]["message"]
    assert runs() == 2
    assert any(l.startswith("tool call does not match its schema, dropped: sparky_enable_tools at categories (type)") for l in _warnings(caplog))


def test_structured_invalid_call_next_to_a_valid_one_is_dropped(client, reply, runs):
    bad = {"text": None, "tool_calls": [
        {"name": "sparky_log_food", "arguments": {"name": "Ei"}},
        {"name": "sparky_enable_tools", "arguments": {"categories": ["goals"], "extra": 1}},
    ]}
    reply({"_queue": [envelope(structured=bad), envelope(structured=bad)]})
    r = client.post(URL, headers=AUTH, json={"tools": TOOLS + [ENABLE_TOOL], "messages": [{"role": "user", "content": "Ei"}]})
    assert _calls(r) == [("sparky_log_food", {"name": "Ei"})]
    assert runs() == 2


def test_structured_arguments_as_text_are_repaired(client, reply, runs):
    reply(envelope(structured={"text": None, "tool_calls": [
        {"name": "sparky_enable_tools", "arguments": '{"categories": "goals"}'},
    ]}))
    r = client.post(URL, headers=AUTH, json={"tools": [ENABLE_TOOL], "messages": [{"role": "user", "content": "x"}]})
    assert _calls(r) == [("sparky_enable_tools", {"categories": ["goals"]})]
    assert runs() == 1


def test_schema_check_also_sees_values_the_coercion_could_not_fix(client, reply, runs):
    reply({"_queue": [
        envelope(structured={"text": None, "tool_calls": [{"name": "sparky_manage_checkin", "arguments": {"action": "log_sleep", "steps": "12.5"}}]}),
        envelope(structured={"text": None, "tool_calls": [{"name": "sparky_manage_checkin", "arguments": {"action": "log_sleep", "steps": 12}}]}),
    ]})
    r = client.post(URL, headers=AUTH, json={"tools": [SLEEP_TOOL], "messages": [{"role": "user", "content": "x"}]})
    assert _calls(r) == [("sparky_manage_checkin", {"action": "log_sleep", "steps": 12})]
    assert runs() == 2


def test_coerce_arguments_repairs_strings_for_arrays():
    from app.translate import coerce_arguments

    schema = ENABLE_TOOL["function"]["parameters"]
    assert coerce_arguments({"categories": '["goals","food"]'}, schema) == {"categories": ["goals", "food"]}
    assert coerce_arguments({"categories": "goals"}, schema) == {"categories": ["goals"]}
    assert coerce_arguments({"categories": "goals, reports"}, schema) == {"categories": ["goals", "reports"]}
    # Not repairable: not an item, one part not an item, broken JSON.
    assert coerce_arguments({"categories": "nutrition"}, schema) == {"categories": "nutrition"}
    assert coerce_arguments({"categories": "goals, nutrition"}, schema) == {"categories": "goals, nutrition"}
    assert coerce_arguments({"categories": '["goals"'}, schema) == {"categories": '["goals"'}
    # A comma list is split only for enum items; a free string becomes one item.
    free = {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    assert coerce_arguments({"tags": "a, b"}, free) == {"tags": ["a, b"]}
    # An object as a JSON string, with its own scalars repaired.
    nested = {"type": "object", "properties": {"sets": {"anyOf": [{"type": "object", "properties": {"reps": {"type": "integer"}}}, {"type": "null"}]}}}
    assert coerce_arguments({"sets": '{"reps": "10"}'}, nested) == {"sets": {"reps": 10}}


def test_argument_error_gives_path_and_keyword_only():
    from app.translate import argument_error

    schema = ENABLE_TOOL["function"]["parameters"]
    assert argument_error({"categories": ["goals"]}, schema) is None
    assert argument_error({}, schema) == ". (required)"
    assert argument_error({"categories": []}, schema) == "categories (minItems)"
    assert argument_error({"categories": ["goals", "x"]}, schema) == "categories/1 (enum)"
    assert argument_error({"categories": ["goals"], "more": 1}, schema) == ". (additionalProperties)"
    assert argument_error("text", schema) == ". (type)"
    nested = {"type": "object", "properties": {"id": {"type": "string", "format": "uuid"}}}
    assert argument_error({"id": "not-a-uuid"}, nested) == "id (format)"
    assert argument_error({"id": "0d3f1a5e-2b8c-4e7a-9f1d-6c2b3a4d5e6f"}, nested) is None
    # A schema the validator can not use is not checked.
    assert argument_error({"x": 1}, {"type": "object", "properties": {"x": {"type": "no-such-type"}}}) is None
    assert argument_error({"x": 1}, "not a schema") is None
