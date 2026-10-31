"""A fake `claude` binary and a test client.

The bridge passes only an allow-list of environment variables to claude, so
the fake binary reads its canned answer from a file whose path is baked into
the script, and writes argv + the stdin message next to it for the asserts.

The fake speaks the stream-json protocol: it reads one user message line
from stdin and prints the events of the canned reply, one JSON line each.
The reply file holds one result event, or a list of events. A list item
`{"_sleep": N}` makes the fake wait N seconds before the next event,
`{"_raw": "..."}` prints that text as a line, and `{"_exit": N}` ends the
fake with exit code N. A single event with `_sleep` waits before it is
printed. `{"_queue": [reply, reply, ...]}` gives one reply per run, in order;
the fake writes the rest back. Each run appends one line to runs.log.
"""

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

import pytest

BASE = Path(tempfile.mkdtemp(prefix="bridge-test-"))
REPLY = BASE / "reply.json"
SEEN = BASE / "seen.json"
RUNS = BASE / "runs.log"
FAKE = BASE / "claude"

FAKE.write_text(
    f"""#!{sys.executable}
import json, sys, os, time
argv = sys.argv[1:]
line = sys.stdin.readline()
message = json.loads(line) if line.strip() else None
rest = sys.stdin.read()
json.dump({{"argv": argv, "message": message, "stdin_rest": rest, "env": dict(os.environ), "cwd": os.getcwd(), "pid": os.getpid()}}, open({str(SEEN)!r}, "w"))
open({str(RUNS)!r}, "a").write("run\\n")
reply = json.load(open({str(REPLY)!r}))
if isinstance(reply, dict) and "_queue" in reply:
    queue = reply["_queue"]
    reply = queue.pop(0)
    json.dump({{"_queue": queue}}, open({str(REPLY)!r}, "w"))
if isinstance(reply, dict):
    reply = [{{"_sleep": reply.pop("_sleep", 0)}}, reply]
for event in reply:
    if set(event) == {{"_sleep"}}:
        time.sleep(event["_sleep"])
        continue
    if set(event) == {{"_raw"}}:
        print(event["_raw"], flush=True)
        continue
    if set(event) == {{"_exit"}}:
        sys.exit(event["_exit"])
    print(json.dumps(event), flush=True)
"""
)
FAKE.chmod(FAKE.stat().st_mode | stat.S_IEXEC)

os.environ.update(
    {
        "CLAUDE_CODE_OAUTH_TOKEN": "test-oauth",
        "BRIDGE_TOKEN": "test-bridge",
        "BRIDGE_CLAUDE_BIN": str(FAKE),
        "BRIDGE_HOME": str(BASE / "home"),
        "BRIDGE_COUNTER_FILE": str(BASE / "counter.json"),
        "BRIDGE_SETTINGS_DIR": str(Path(__file__).resolve().parent.parent / "settings"),
        "BRIDGE_DAILY_LIMIT": "50",
    }
)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

AUTH = {"Authorization": "Bearer test-bridge"}


def envelope(result="", structured=None, is_error=False):
    env = {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "result": result,
        "duration_ms": 10,
        "duration_api_ms": 8,
        "num_turns": 1,
        "usage": {"input_tokens": 100, "cache_read_input_tokens": 20, "output_tokens": 7},
    }
    if structured is not None:
        env["structured_output"] = structured
    return env


def assistant_event(blocks, usage=None):
    """One `assistant` stream-json event with the given content blocks."""
    message = {"role": "assistant", "content": blocks}
    if usage is not None:
        message["usage"] = usage
    return {"type": "assistant", "message": message}


def tool_use(name, arguments):
    return {"type": "tool_use", "id": "toolu_1", "name": name, "input": arguments}


@pytest.fixture
def reply():
    def set_reply(env):
        REPLY.write_text(json.dumps(env))

    if (BASE / "counter.json").exists():
        (BASE / "counter.json").unlink()
    if SEEN.exists():
        SEEN.unlink()
    if RUNS.exists():
        RUNS.unlink()
    return set_reply


@pytest.fixture
def seen():
    def read():
        return json.loads(SEEN.read_text()) if SEEN.exists() else None

    return read


@pytest.fixture
def runs():
    """How many times the fake claude ran since the reply was set."""

    def count():
        return len(RUNS.read_text().splitlines()) if RUNS.exists() else 0

    return count


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_cache_state():
    from app import main

    main._cache_state["ttl"] = "1h"
    yield
