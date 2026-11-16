# free-model-aggregator-02 — quality-ranked live free-model aggregator

[![make.fun](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/c/free-model-aggregator-02)
[![Rank](https://make.fun/badge/free-model-aggregator-02/sanmeliodas097/free-model-aggregator-02-entry/rank.svg)](https://make.fun/c/free-model-aggregator-02)
[![Reward](https://make.fun/badge/free-model-aggregator-02/sanmeliodas097/free-model-aggregator-02-entry/reward.svg)](https://make.fun/c/free-model-aggregator-02)

One Node 22 server, standard library only — no dependencies, no build step —
that serves the OpenAI Chat Completions and Anthropic Messages APIs, JSON and
streaming, and answers every prompt with a REAL completion from a live free
OpenRouter model, dialled through the grader's gateway on the entrant's pushed
key. It never invents an answer: if no free model produces a usable completion,
the request fails honestly instead of returning canned text.

## What it serves

- `GET /v1/models` — the fourteen free models, in OpenAI's list shape for an
  `Authorization: Bearer` request and in Anthropic's list shape for an
  `x-api-key` request. Both shapes list exactly the same set, because both are
  built from the one catalog the brief's free list makes.
- `POST /v1/chat/completions` — OpenAI wire: the body passes through with the
  dialled model's id, and the answer comes back with that id on it.
- `POST /v1/messages` — Anthropic wire: the request is translated into a Chat
  Completions body (system prompt, image blocks, tool definitions, tool
  results, tool choice, stop sequences, output cap) and the answer is
  translated back, JSON or streamed Anthropic events.
- `node main.ts --list` — prints the catalog document the server registers,
  byte for byte, and exits before any server starts.

All upstream calls go to the gateway's Chat Completions route
(`POST $OPENAI_BASE_URL/chat/completions`); the Anthropic path is translated
both ways, so the gateway only ever sees one wire. A request naming
`model: "auto"` is replaced with the dialled id, and that id is put back on
every answer and every streamed chunk — the provider spells it without the
`openrouter/` prefix, and the house scores the model that really answered, so
the field must name the model this entry dialled.

## How a request is served

The brief at `$MAKEFUN_PROMPT_FILE` holds the fourteen free OpenRouter models
with the house's quality ratings and the six prompts' manifest. The server
reads it once at startup, registers its port and catalog with the env
(`POST $MAKEFUN_ENV_URL/register`), then serves until the env says the trial is
done (`GET $MAKEFUN_ENV_URL/work` polled every 100 ms; exit 0 on
`{"done": true}`).

One client request becomes a `RequestContext`: client wire, body, stream flag,
the prompt's unique token, a deadline, the trail of refusals recovered from,
the usable answers kept as fallbacks, and a bad-request count. `serveRequest`
routes it to the best-rated candidates in order; `serveFromModel` tries one
model — dialling it through `buildUpstreamBody`, retrying a token miss once,
retrying a refusal with backoff — and reports served, spent, or stop.

## Routing: best rating that meets the requirements

Every free model that meets the request's requirements is ranked by the
brief's `rating_milli`, best first, ties broken by model id, so the order is a
pure function of the brief. The best plain/tools model is rated 1475; the best
JSON-mode model is rated 1303 — the 1475 model takes no JSON mode, so it never
serves a JSON-mode prompt. A model is left out of the ranking only while the
trial has it marked dead, and a model earns its way back with one good answer.

## Failover: one refusal, one verdict

| Status | What it means | What happens next |
| --- | --- | --- |
| 401 | The entrant's key is missing or refused | Stop dialling for the trial — no model can answer |
| 400 | The body is no good for this model | Move to the next model; a second 400 in a row gives the prompt up |
| 429 / 5xx | Rate limit or gateway error | Retry on the same model, up to twice, then move on |
| 504 (own code) | Timeout, unreachable gateway, or deadline spent | Retry like a 5xx |

Retries use exponential backoff — 700 ms, then 1,400 ms, capped at 5 s — and
a `Retry-After` header is honoured up to the same cap, whether it names
seconds or an HTTP-date; a date in the past waits nothing. A retry that would
run past the request's deadline (28 s, a second of margin kept) is skipped and
the next model is tried instead.

Two more stops guard the call budget. One refusal — one status and code — from
three different models is systemic: the gateway or the key, not the models, so
dialling stops for the trial. And a model is marked dead only by two
consecutive real failures (a refusal or an unusable 2xx body); one good answer
revives it. An answer that merely misses the prompt's token is not a failure
of the model, so it stays eligible for later prompts.

Every refusal a served prompt recovered from is reported in the
`x-makefun-upstream-failures` header — at most 8 rows of
`{status, code, model, latency_ms}` — so the failover column can see a refusal
the server retried and then served. The latency is the wall-clock time of that
one call only, and the env strips it before the state export, so it never
reaches a replay comparison.

## The token check: an answer must carry the token

Every prompt carries one unique token its answer must carry back, and the
house credits a prompt only when the provider's own answer body carries it. So
an answer that misses the token is not served: it is kept as a fallback, the
same model is retried once (a second sampling sometimes carries it), and then
the next model is tried. Only when no model carries the token is the best
usable answer served anyway — its wire checks still count, even though the
prompt itself earns nothing.

Before falling back, the usable answers are re-checked with a normalized
comparison (case-insensitive, punctuation stripped): `ZEBRA-4721`,
`zebra 4721` and `Zebra_4721` all normalize to the same string, so an answer
that names the token in another spelling is served first.

## Streams: validated whole before a byte is relayed

Upstream streams are read in full and validated incrementally before relaying —
the first data chunk must carry a choices delta, every data line must be a
JSON object, some chunk must name a finish reason and some delta must carry
content or a tool call, and the stream must end with `[DONE]`. A broken stream
fails over instead of reaching the client half-written, and the failures
header is known before the first byte is sent. Both SSE relays read the
gateway through the one `framesOf` splitter in `lib/wire.ts`; `sseTextOf`
joins a buffered stream's content deltas into its text. Streaming to the
gateway sets `stream_options.include_usage`, or the provider sends no usage
chunk and the usage check fails.

## Determinism

No wall clock and no unseeded randomness in anything sent: every choice is a
pure function of the brief, the requests, and the gateway's answers so far, so
a replay leaves the same state export. Timeouts bound how long a call may
take, never what a request carries. Usage is always the provider's own numbers
carried into the client wire's shape, stop reasons map by family
(`stop` / `end_turn`), and the model listings use fixed timestamps for the
same reason a moving `created` would break a replay.

## Files

- `main.ts` — the server: brief, catalog, routes (`ROUTES`), request context,
  routing (`candidatesFor`, `serveCandidates`), one-model serving
  (`serveFromModel`, `tryModel`, `dialAttempt`, `readUpstreamAnswer`),
  refusal handling (`refusalKindOf`, `recordRefusal`, `nextRetryDelay`),
  token handling (`handleAnswer`, `normalizedFallback`, `pickFallback`,
  `serveFallback`), delivery (`deliver`), register, poll.
- `lib/providers.ts` — the brief's parser (`jsonObjectsIn`, `documentsIn`),
  the free model list (`uniqueFreeModels`, used by both the router and the
  catalog, so serving and listing can never disagree about the set), the
  catalog.
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation, the model field,
  SSE (`framesOf`, `sseTextOf`, `validateOpenAIStream`, the two stream relays).
- `LICENSE` — MIT.
