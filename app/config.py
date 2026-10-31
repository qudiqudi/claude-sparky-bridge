"""Settings from the environment. Read once at import."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


@dataclass(frozen=True)
class Config:
    oauth_token: str = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    bridge_token: str = os.environ.get("BRIDGE_TOKEN", "")
    claude_bin: str = os.environ.get("BRIDGE_CLAUDE_BIN", "/opt/claude/claude")
    home: str = os.environ.get("BRIDGE_HOME", "/data/home")
    counter_file: str = os.environ.get("BRIDGE_COUNTER_FILE", "/data/counter.json")
    settings_dir: str = os.environ.get("BRIDGE_SETTINGS_DIR", "/app/settings")
    default_model: str = os.environ.get("BRIDGE_MODEL", "sonnet")
    classifier_model: str = os.environ.get("BRIDGE_CLASSIFIER_MODEL", "haiku")
    effort: str = os.environ.get("BRIDGE_EFFORT", "low")
    timeout_s: int = _int("BRIDGE_TIMEOUT_S", 240)
    classifier_timeout_s: int = _int("BRIDGE_CLASSIFIER_TIMEOUT_S", 9)
    daily_limit: int = _int("BRIDGE_DAILY_LIMIT", 300)
    max_prompt_bytes: int = _int("BRIDGE_MAX_PROMPT_BYTES", 900_000)
    # One 12 MB image as base64 (16 MB) plus the chat history.
    max_body_bytes: int = _int("BRIDGE_MAX_BODY_BYTES", 25_000_000)
    max_image_bytes: int = _int("BRIDGE_MAX_IMAGE_BYTES", 12_000_000)
    max_total_image_bytes: int = _int("BRIDGE_MAX_TOTAL_IMAGE_BYTES", 16_000_000)
    max_images: int = _int("BRIDGE_MAX_IMAGES", 4)
    # Parallel claude processes, and how many requests may wait for one.
    max_concurrent: int = _int("BRIDGE_MAX_CONCURRENT", 2)
    max_queued: int = _int("BRIDGE_MAX_QUEUED", 8)
    # Cache mark on the tool list of a chat request, so that the next request
    # reads it from the prompt cache. A run that claude refuses because of the
    # mark goes once more without it (see main._run_plan).
    cache_tools: bool = _bool("BRIDGE_CACHE_TOOLS", True)

    @property
    def configured(self) -> bool:
        return bool(self.oauth_token and self.bridge_token and self.daily_limit > 0)


cfg = Config()

# Model names the bridge passes through to `claude --model`. Anything else
# (for example a name typed in the SparkyFitness form) falls back to
# BRIDGE_MODEL, so a typo can not break the chat.
ALLOWED_MODELS = {"sonnet", "opus", "haiku", "fable"}


def resolve_model(requested: object) -> str:
    name = requested.strip() if isinstance(requested, str) else ""
    if name in ALLOWED_MODELS or name.startswith("claude-"):
        return name
    return cfg.default_model
