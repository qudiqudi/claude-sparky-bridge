# claude-sparky-bridge: OpenAI-compatible endpoint for SparkyFitness, backed by Claude Code.
FROM python:3.12.13-slim

# PINNED Claude Code version, auto update off. 2.1.316 is the tested version:
# it has --json-schema, structured_output, and stream-json input with image
# blocks, which the bridge depends on.
ARG CLAUDE_VERSION=2.1.316

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    TZ=UTC \
    DISABLE_AUTOUPDATER=1 \
    BRIDGE_CLAUDE_BIN=/opt/claude/claude

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates tzdata \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# The install runs with its own HOME, so the runtime HOME (/data/home, a
# volume) never shadows the binary. The version check fails the BUILD if the
# installer gives a different version.
RUN mkdir -p /opt/claude \
 && HOME=/opt/claude DISABLE_AUTOUPDATER=1 \
      bash -c "set -euo pipefail; curl -fsSL https://claude.ai/install.sh | bash -s ${CLAUDE_VERSION}" \
 && ln -sf /opt/claude/.local/bin/claude /opt/claude/claude \
 && test -x /opt/claude/claude \
 && test "$(/opt/claude/claude --version | awk '{print $1}')" = "${CLAUDE_VERSION}"

COPY settings /app/settings
COPY app /app/app

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin bridge \
 && mkdir -p /data/home \
 && chown -R bridge:bridge /data /opt/claude
USER bridge

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/healthz >/dev/null || exit 1

# One worker: the daily counter file is not shared between processes.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
