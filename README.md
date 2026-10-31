# claude-sparky-bridge

This bridge lets [SparkyFitness](https://github.com/CodeWithCJ/SparkyFitness) use your Claude subscription as its AI service. You do not pay for an API key.

SparkyFitness has an AI chat. The chat can log meals, estimate food from a photo, and read nutrition labels. It can also tell you about your data. For this, SparkyFitness must have an AI service.

The bridge is a small container. It gives SparkyFitness an API in the OpenAI format. It sends each request to Claude Code, which uses your Claude Pro or Max plan. You do not pay a cost for each token.

## What the bridge does

- Chat with tools. Claude can use the SparkyFitness tools, for example to log food or to change goals. SparkyFitness runs the tools. Thus, the user permissions and the meal cards do not change.
- Tool call check. The bridge compares the arguments of each tool call with the schema of the tool. It repairs a number or a list that Claude wrote as text. If a call does not match the schema, Claude gets one more try. The bridge does not send a call that does not match to SparkyFitness.
- Food photos and label scans. The bridge gives the image to Claude in the message. It sends the result back in the format that SparkyFitness uses.
- Connection test. The Test button in SparkyFitness gets an answer immediately, without a call to Claude.
- Daily limit. A counter stops the bridge after a specified number of calls each day. The default is 300.

## Security

- The bridge has no published port. Only containers in the Docker network of SparkyFitness can reach it.
- Each request must have the bridge token.
- All Claude Code tools are off, for each request. Claude can not read files, run commands, or use the web.
- A photo goes to Claude inside the message. The bridge does not write it to disk.
- The bridge has limits for the size of a request, the size and number of images, and the number of parallel Claude processes. See Settings.
- The log shows the type of request, the model, the duration, and the token counts, with the cache counts. It does not show your messages or photos.
- If Claude Code stops with an error, the log shows the first 160 characters of that error.
- If a tool call does not match the schema of the tool, the log shows the name of the tool and the position in the schema. It does not show the values.

## Requirements

- SparkyFitness v1.11.0 or a subsequent version, in Docker. In earlier versions, all AI requests stop with the error `fetch failed`.
- Docker with Docker Compose v2.
- A Claude Pro or Max subscription.
- Claude Code on a computer, one time, to make the token.

## Install

1. On a computer with Claude Code, make a long-term token. Keep the token for step 5.

   ```
   claude setup-token
   ```

2. On your Docker host, get the code.

   ```
   git clone https://github.com/qudiqudi/claude-sparky-bridge.git
   cd claude-sparky-bridge
   cp .env.example .env
   ```

3. Make a random bridge token.

   ```
   openssl rand -hex 32
   ```

4. Find the Docker network of the SparkyFitness server container. The command shows the container name and its networks.

   ```
   docker ps --filter name=sparkyfitness-server --format '{{.Names}}: {{.Networks}}'
   ```

5. In `.env`, set `CLAUDE_CODE_OAUTH_TOKEN`, `BRIDGE_TOKEN`, and `SPARKY_NETWORK`.

6. Start the bridge.

   ```
   docker compose up -d --build
   ```

7. In SparkyFitness, go to Settings, AI Service Settings, Add New AI Service.

8. Type these values:

   - Service type: OpenAI Compatible
   - Custom URL: `http://claude-sparky-bridge:8080/v1`
   - API Key: the value of `BRIDGE_TOKEN`
   - Model: `sonnet`
   - Tool profile: `full`

9. Click Save, then click Test.

10. Select the service as the active service for chat and for vision.

You can also put the bridge into the compose file of SparkyFitness:

1. Put the `bridge` service and the `bridge-data` volume from `docker-compose.yml` into that file.
2. Change `build` to the path of this folder.
3. Change the `networks` of the bridge to the network of your SparkyFitness services.
4. Put `CLAUDE_CODE_OAUTH_TOKEN` and `BRIDGE_TOKEN` into the `.env` file of SparkyFitness. Compose does not read the `.env` file of this folder.

## Settings

All settings are environment variables in `.env`.

- `CLAUDE_CODE_OAUTH_TOKEN`: the token from `claude setup-token`. This value is necessary.
- `BRIDGE_TOKEN`: the secret that the bridge and SparkyFitness use. This value is necessary.
- `SPARKY_NETWORK`: the Docker network of SparkyFitness. This value is necessary.
- `BRIDGE_MODEL`: the model for chat and photos. The default is `sonnet`. The bridge accepts `sonnet`, `opus`, `haiku`, `fable`, and full `claude-*` names. Other names use the default.
- `BRIDGE_CLASSIFIER_MODEL`: the fast model for the topic step of SparkyFitness. SparkyFitness does this step only when its keyword check finds no topic. The default is `haiku`.
- `BRIDGE_EFFORT`: the effort level of Claude Code. The default is `low`.
- `BRIDGE_DAILY_LIMIT`: the maximum number of Claude calls each day. The default is `300`. After the limit, the bridge sends HTTP 429. For a stream request, it sends an error event.
- `BRIDGE_TIMEOUT_S`: the maximum time for one call, in seconds. The default is `240`.
- `TZ`: the time zone. The default is `UTC`. The daily counter starts again at midnight in this zone.

These settings limit the size of a request and the number of parallel Claude processes. A request that is too large gets HTTP 413.

- `BRIDGE_MAX_BODY_BYTES`: the maximum size of one request body, in bytes. The default is `25000000`. This is sufficient for one image of 12 MB as base64 and the chat history.
- `BRIDGE_MAX_IMAGE_BYTES`: the maximum size of one image, in bytes. The default is `12000000`.
- `BRIDGE_MAX_TOTAL_IMAGE_BYTES`: the maximum size of all images in one request, in bytes. The default is `16000000`.
- `BRIDGE_MAX_IMAGES`: the maximum number of images in one request. The default is `4`.
- `BRIDGE_MAX_PROMPT_BYTES`: the maximum size of the text that goes to Claude, in bytes. The default is `900000`.
- `BRIDGE_MAX_CONCURRENT`: the maximum number of Claude processes at the same time. The default is `2`. More requests wait.
- `BRIDGE_MAX_QUEUED`: the maximum number of requests that wait for a free process. The default is `8`. More requests get HTTP 429 immediately.
- `BRIDGE_CACHE_TOOLS`: puts a cache mark on the tool list of each chat request. The next request can then read the tool list from the prompt cache, which can make the answer faster. The default is `1`. Set `0` or `false` to turn the mark off. Claude Code sets its own cache marks, and the lifetime of these marks can be 1 hour or 5 minutes. If Claude Code refuses the mark, the bridge tries a mark with 1 hour, then with 5 minutes, then no mark. It keeps the choice that operates, and writes a warning to the log.

## Status

To see the status, use this command.

```
docker exec claude-sparky-bridge curl -s http://127.0.0.1:8080/healthz
```

The result shows `configured`, `calls_today`, `daily_limit`, `running`, and `waiting`.

## Limits

- Each request starts one Claude Code process. If Claude Code refuses the cache mark, the bridge can start up to two more. If a tool call does not match the schema of the tool, it can start one more.
- A chat answer is ready in some seconds. A message with a photo is ready in approximately 15 to 25 seconds.
- The answer comes as one block. It does not come in parts.
- Keep the bridge token secret. Each request with a valid token uses your subscription.
- Each request uses the usage limits of your subscription, the same as Claude Code on your computer.
- Use the bridge only for your account. Obey the [Anthropic terms](https://www.anthropic.com/legal/consumer-terms) for your plan.

## Tests

The tests use a fake `claude` program. They do not use a token.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/
```

## License

MIT
