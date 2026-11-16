// free-model-aggregator-02: one server, two client wires, live free models behind every answer.
//
// The server reads the brief at $MAKEFUN_PROMPT_FILE (the fourteen free OpenRouter models with
// the house's ratings, and the six prompts' manifest), then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header
//      it got: x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions
//      and POST /v1/messages, each with a JSON body or with server-sent events;
//   4. answers each prompt by dialling ONE free model through the gateway's Chat Completions route
//      (POST $OPENAI_BASE_URL/chat/completions, Authorization: Bearer $OPENAI_API_KEY, the trial's
//      dummy key the gateway swaps for the entrant's pushed OpenRouter key). An Anthropic-shaped
//      request is translated into a Chat Completions body first, and the answer translated back;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//      A process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts;
// the register body's catalog must be that same document. Build it once and use the one object.
//
// ROUTING (quality is 60% of the score): for each request, every free model that meets the
// request's requirements is ranked by the brief's rating_milli, best first, ties by model id.
// The best plain/tools model is the 1475-rated one; the best JSON-mode model is the 1303-rated
// one (the 1475 model takes no JSON mode, so it never serves a JSON-mode prompt).
//
// FAILOVER (10%): a gateway call answered 429 or 5xx is retried up to twice on the same model,
// with exponential backoff (700 ms, then 1,400 ms, capped at 5 s; a Retry-After the gateway
// sends is honoured up to that same cap), then the request moves to the next-best free model
// that meets it. A 400 moves straight to the next model (the body is no good for this one); two
// 400s in a row give the prompt up. A 401 means the entrant's key is missing or refused and no
// model can answer, so the server stops dialling for the rest of the trial instead of burning
// the call budget. The same holds when one refusal (one status and code) comes back from three
// different models: that is the gateway or the key, not the models, so further calls are
// pointless. Every refusal a served prompt recovered from is reported in the
// `x-makefun-upstream-failures` header (at most 8 rows of {status, code, model, latency_ms}):
// without it the failover column cannot see a refusal the server retried and then served. The
// latency is the wall-clock time of that one call only, and the env strips it before the state
// export, so it never reaches a replay comparison. A model is marked dead only by two
// consecutive real failures (a refusal or an unusable 2xx body), and one good answer revives it:
// an answer that merely misses the prompt's token is not a failure of the model, so it stays
// eligible for later prompts.
//
// THE TOKEN CHECK: every prompt carries one unique token its answer must carry back, and the
// house credits a prompt only when the provider's own answer body carries it. So an answer
// that does not carry the token is not served: the next-best model is tried, and only when no
// model produces the token does the server serve the best usable answer it got (its wire
// checks still count, even though the prompt itself earns nothing). A token miss is worth one
// retry on the same model first — a refusal or a rambling first answer sometimes carries the
// token on a second sampling — and only then does the request move to the next model. Before
// falling back, the usable answers are re-checked with a normalized comparison
// (case-insensitive, punctuation stripped): an answer that carries the token in another
// spelling is the best fallback, since it at least names the token the prompt asked for.
//
// THE WIRES: the gateway speaks Chat Completions only. OpenAI client bodies pass through with
// the dialled model id; Anthropic client bodies are translated both ways (system, tools,
// images, tool results, stop sequences, the output cap). Usage is always the provider's own
// numbers carried into the client wire's shape; stop reasons map by family (stop / end_turn).
// The provider's answer names the model without the gateway's `openrouter/` prefix: the id
// dialled is put back on every answer and every streamed chunk. Upstream streams are read in
// full and validated incrementally before anything is relayed — the first data chunk must carry
// a choices delta, every data line must be a JSON object, some chunk must name a finish reason
// and some delta must carry content or a tool call, and the stream must end with [DONE] — so a
// broken stream fails over instead of reaching the client half-written, and the failures header
// is known before the first byte is sent.
// Streaming to the gateway sets stream_options.include_usage, or the provider sends no usage
// chunk and the usage check fails.
//
// DETERMINISM: no wall clock and no unseeded randomness in anything sent. Every choice is a
// pure function of the brief, the requests and the gateway's answers so far, so a replay
// leaves the same state export. Timeouts bound how long a call may take, never what a request
// carries: a call that is still unanswered when its time is up is abandoned, and the next
// model is tried.
//
// STRUCTURE: one client request becomes a RequestContext (contextOf: client wire, body, stream
// flag, token, deadline, trail rows, fallbacks, bad-request count). serveRequest routes it to the
// best-rated candidates in order through serveCandidates; serveFromModel tries one model —
// dialling it through buildUpstreamBody, retrying a token miss once (handleAnswer), retrying a
// refusal with backoff — and reports served, spent, or stop.
// One dial is split in three: dialAttempt POSTs and turns a transport failure into a refusal,
// readUpstreamAnswer reads the response in full and classifies it, and nextRetryDelay decides
// whether a retry fits the budget and the deadline (retryDelayMs over the exponential
// backoffDelayMs, honouring a Retry-After up to the same cap). refusalKindOf classifies each
// refusal by status family (see the REFUSAL DECISION TABLE at recordRefusal: 401 stops the
// trial, 400 moves on and gives up after two, a 429/5xx/timeout retries unless one refusal has
// come from three models, which is systemic); pickFallback chooses the best usable answer when
// no model carried the token, preferring one that carries it under the normalized comparison
// (normalizedFallback), and serveFallback serves it as the request's last resort. The POST
// routes live in the ROUTES table in handle().

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  sseTextOf,
  type ServedModel,
  validateOpenAIStream,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** How long one gateway call may take, in milliseconds, from request to last byte. */
const UPSTREAM_TIMEOUT_MS = 20000;
/** How long one client request may take in all, in milliseconds (the client caps at 30 s). */
const REQUEST_DEADLINE_MS = 28000;
/** The pause before the first retry of a refused call on the same model, in milliseconds. */
const RETRY_BASE_DELAY_MS = 700;
/** The longest pause before a retry, in milliseconds: the backoff doubles to this cap. */
const RETRY_MAX_DELAY_MS = 5000;
/** How many times a 429/5xx/timeout is retried on the same model before moving on. */
const MAX_MODEL_RETRIES = 2;
/** The most refusal rows the failures header carries. */
const MAX_TRAIL_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** True once the gateway has answered 401: the entrant's key is missing or refused, and no model can answer. */
let keyDead = false;
/** True once one refusal has come back from three different models: the gateway or the key, not the models. */
let gatewayDead = false;
/** The models each refusal (one status and code) has come back from, this trial. */
const refusalModels = new Map<string, Set<string>>();

/** Whether this refusal is systemic: the same status and code from three different models. */
function systemicRefusal(status: number, code: string, model: string): boolean {
  const key = `${status}:${code}`;
  let set = refusalModels.get(key);
  if (set === undefined) {
    set = new Set<string>();
    refusalModels.set(key, set);
  }
  set.add(model);
  return set.size >= 3;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** One model's health this trial: consecutive failures, and whether it is spent for good. */
interface ModelState {
  fails: number;
  dead: boolean;
}
const modelState = new Map<string, ModelState>();

function stateOf(modelId: string): ModelState {
  let state = modelState.get(modelId);
  if (state === undefined) {
    state = { fails: 0, dead: false };
    modelState.set(modelId, state);
  }
  return state;
}

/** A model answered well: its failure streak resets, and it is alive again — a model the trial
 * marked dead earns its way back with one good answer, so a rate limit that cleared does not
 * cost the trial its best model for the remaining prompts. */
function noteSuccess(modelId: string): void {
  const state = stateOf(modelId);
  state.fails = 0;
  state.dead = false;
}

/** A model failed: count it, and mark it dead on the second consecutive failure. */
function noteFailure(modelId: string): void {
  const state = stateOf(modelId);
  state.fails += 1;
  if (state.fails >= 2) state.dead = true;
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that can serve one request, best rating first. Ties break by model id, so the
 * order is a pure function of the brief. A model the trial has marked dead is left out.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const out = freeModels(brief).filter((model) => meetsRequirements(requires, capabilitiesOf(model)));
  out.sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return out.filter((model) => !stateOf(model.id).dead);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Every text the request carries, in order: the system prompt, then each message's text. */
function requestTexts(body: Json): string[] {
  const texts: string[] = [];
  const system = typeof body.system === 'string' ? body.system : '';
  if (system !== '') texts.push(system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const content = message.content;
    if (typeof content === 'string') texts.push(content);
    else if (Array.isArray(content)) {
      for (const block of content) {
        if (isRecord(block) && typeof block.text === 'string') texts.push(block.text);
      }
    }
  }
  return texts;
}

/** The prompt's unique token, read off its user message: the one token-shaped word in it. */
function tokenOf(body: Json): string | null {
  const match = /[A-Z]{2,}-[0-9]{4}/.exec(requestTexts(body).join(' '));
  return match === null ? null : match[0];
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extraHeaders: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type, null), extraHeaders);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The failures header for a response that recovered from refusals: at most 8 rows, in order. */
function trailHeader(failures: TrailRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_TRAIL_ROWS)) };
}

/** One refusal this trial, as the failures header spells it. `latency_ms` is the wall-clock time
 * that one call took; the env reads only status, code and model from the header, so the latency
 * never reaches the state export a replay must reproduce. */
interface TrailRow {
  status: number;
  code: string;
  model: string | null;
  latency_ms: number;
}

/** The error code of a gateway error body, as the env reads it: error.code, else error.type, else ''. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: no code.
  }
  return '';
}

/** The text of a Chat Completions answer: the message content, or its text blocks joined. */
function openAITextOf(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
      .join('');
  }
  return '';
}

/** Whether a Chat Completions JSON answer is well-formed enough to translate. */
function validOpenAIJson(body: Json): boolean {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  return choices.length > 0 && isRecord(choices[0]) && isRecord(choices[0].message);
}

/** A readable stream of one buffered string, for the wire translators that read streams. */
function streamOf(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

/** The pause between a refusal and its retry, in milliseconds. */
function pause(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * The pause before retry number `attempt` (0-based) of a refused call: the exponential backoff,
 * doubled per retry and capped at the maximum. A Retry-After the gateway sent wins when it is
 * shorter than the backoff, so a provider that names its own wait is obeyed; it is capped at the
 * same maximum, so a wild header cannot stall the prompt past its deadline. Retry-After may be a
 * number of seconds or an HTTP-date: a date in the past waits nothing, a date past the cap
 * waits the cap, and a value that is neither waits the backoff.
 */
function backoffDelayMs(attempt: number): number {
  return Math.min(RETRY_BASE_DELAY_MS * 2 ** attempt, RETRY_MAX_DELAY_MS);
}

function retryDelayMs(attempt: number, retryAfter: string | null): number {
  const backoff = backoffDelayMs(attempt);
  if (retryAfter === null) return backoff;
  const header = retryAfter.trim();
  const seconds = Number(header);
  if (Number.isFinite(seconds) && seconds >= 0) return Math.min(Math.floor(seconds * 1000), RETRY_MAX_DELAY_MS, backoff);
  const at = Date.parse(header);
  if (Number.isFinite(at)) return Math.min(Math.max(0, at - Date.now()), RETRY_MAX_DELAY_MS, backoff);
  return backoff;
}

/**
 * Text with case folded and punctuation stripped, for the token re-check: `ZEBRA-4721`,
 * `zebra 4721` and `Zebra_4721` all normalize to the same string, so an answer that spells the
 * token differently is still recognised as naming it.
 */
function normalizedOf(text: string): string {
  return text.toUpperCase().replace(/[^A-Z0-9]/g, '');
}

/** Whether `text` carries `token` under the normalized comparison. */
function normalizedCarries(text: string, token: string): boolean {
  const needle = normalizedOf(token);
  return needle !== '' && normalizedOf(text).includes(needle);
}

/** The answer text of a dialled answer: its JSON content, or its streamed deltas joined. */
function answerTextOf(answer: DialledAnswer): string {
  if (answer.json !== null) return openAITextOf(answer.json);
  return answer.streamText === null ? '' : sseTextOf(answer.streamText);
}

/** One gateway call: POST the Chat Completions body naming `model`, within the time left. */
async function dial(model: string, body: Json, timeoutMs: number): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(Math.max(1, Math.floor(timeoutMs))),
  });
}

/** One usable gateway answer, buffered whole, with whether its text carries the prompt's token. */
interface DialledAnswer {
  model: string;
  json: Json | null;
  streamText: string | null;
  tokenOk: boolean;
}

/** What one model attempt returned: a usable answer, a refusal, or a broken 2xx. A refusal
 * carries the call's latency and any Retry-After the gateway sent, for the trail and the retry. */
type Attempt =
  | { kind: 'answer'; answer: DialledAnswer }
  | { kind: 'refused'; status: number; code: string; latencyMs: number; retryAfter: string | null }
  | { kind: 'broken'; reason: string };

/** Everything one client request carries through routing and failover. */
interface RequestContext {
  clientApi: 'openai' | 'anthropic';
  body: Json;
  stream: boolean;
  token: string | null;
  deadline: number;
  failures: TrailRow[];
  fallbacks: DialledAnswer[];
  badRequests: number;
}

/**
 * The Chat Completions body one model is dialled with: the client's OpenAI body as it came, or
 * the Anthropic body translated, plus the streaming flags the gateway needs.
 */
function buildUpstreamBody(ctx: RequestContext, model: string): Json {
  const upstreamBody: Json = ctx.clientApi === 'openai' ? { ...ctx.body } : anthropicToOpenAI(ctx.body, model);
  if (ctx.stream) {
    upstreamBody.stream = true;
    upstreamBody.stream_options = { include_usage: true };
  }
  return upstreamBody;
}

/**
 * Try one model once for one request: dial it, read the answer in full, and say whether it is
 * usable. The dial itself is dialAttempt; reading and classifying what came back is
 * readUpstreamAnswer. A refusal is classified by its status; a 2xx whose body is unusable is
 * 'broken', not a refusal, and leaves no trail row. The latency is measured from the dial to
 * the last byte read, so the trail can say how long the refused call took.
 */
async function tryModel(ctx: RequestContext, model: string): Promise<Attempt> {
  const remaining = ctx.deadline - Date.now();
  if (remaining < 1000) return { kind: 'refused', status: 504, code: 'deadline', latencyMs: 0, retryAfter: null };
  const startedAt = Date.now();
  const dialled = await dialAttempt(ctx, model, Math.min(UPSTREAM_TIMEOUT_MS, remaining), startedAt);
  if (dialled.kind === 'refused') return dialled;
  return readUpstreamAnswer(ctx, model, dialled.response, startedAt);
}

/**
 * One dial of the gateway for one model: build the upstream body and POST it within the time
 * left. A dial that never reaches the gateway, or runs out of time, is a transient refusal:
 * the caller retries it with backoff like any other 429/5xx, it is not a broken answer.
 */
async function dialAttempt(
  ctx: RequestContext,
  model: string,
  timeoutMs: number,
  startedAt: number,
): Promise<{ kind: 'response'; response: Response } | Extract<Attempt, { kind: 'refused' }>> {
  const upstreamBody = buildUpstreamBody(ctx, model);
  try {
    const response = await dial(model, upstreamBody, timeoutMs);
    return { kind: 'response', response };
  } catch (err) {
    // The gateway could not be reached, or the call ran out of time: a transient failure.
    log(`call to ${model} failed: ${(err as Error).message}`);
    return {
      kind: 'refused',
      status: 504,
      code: (err as Error).name === 'TimeoutError' ? 'timeout' : 'upstream_unreachable',
      latencyMs: Date.now() - startedAt,
      retryAfter: null,
    };
  }
}

/**
 * ANSWER-CLASSIFICATION DECISION TABLE — one buffered 2xx body, by wire:
 *
 *   wire    | body check                                | verdict
 *   --------|-------------------------------------------|---------------------------
 *   JSON    | parses and holds a choices[0].message     | answer (tokenOk by text)
 *   JSON    | unparseable, or no Chat Completions shape | broken (no trail row)
 *   stream  | passes validateOpenAIStream               | answer (tokenOk by deltas)
 *   stream  | fails any stream check ([DONE], finish,   | broken (reason = the check's)
 *           | first delta, content, JSON lines)         |
 *
 * A refusal (non-2xx) never reaches this table: it is classified in
 * readUpstreamAnswer, which also owns the two checks both wires share — an
 * empty body is broken, and the latency is measured from the dial to the last
 * byte read. Classification is separate from the verdict's side effects
 * (noteSuccess / noteFailure / the token retry in handleAnswer and
 * serveFromModel), so this table is the one place that says what a 2xx body is.
 */
function classifyJsonAnswer(ctx: RequestContext, model: string, text: string): Attempt {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { kind: 'broken', reason: 'unparseable JSON' };
  }
  if (!isRecord(parsed) || !validOpenAIJson(parsed)) return { kind: 'broken', reason: 'not a Chat Completions answer' };
  return {
    kind: 'answer',
    answer: { model, json: parsed, streamText: null, tokenOk: ctx.token === null ? true : openAITextOf(parsed).includes(ctx.token) },
  };
}

/**
 * Classify one buffered upstream stream: it is an answer only when the whole
 * stream validates — the first chunk carries a choices delta, every data line
 * is one JSON object, some chunk names a finish reason, some delta carries
 * content or a tool call, and the stream ends with [DONE]. Anything less is
 * 'broken' with the failing check's reason, so the caller fails over instead
 * of relaying half an answer. The token is read off the joined content
 * deltas, the same text the client will receive.
 */
function classifyStreamAnswer(ctx: RequestContext, model: string, text: string): Attempt {
  const streamCheck = validateOpenAIStream(text);
  if (!streamCheck.ok) return { kind: 'broken', reason: streamCheck.reason };
  return {
    kind: 'answer',
    answer: { model, json: null, streamText: text, tokenOk: ctx.token === null ? true : sseTextOf(text).includes(ctx.token) },
  };
}

/**
 * Read one dialled response in full and say whether it is usable. A refused status becomes a
 * refusal carrying the call's latency and any Retry-After the gateway sent, for the trail and
 * the retry; a 2xx whose body is empty, unparseable, or not a Chat Completions answer is
 * 'broken'. The body itself is classified by wire — classifyJsonAnswer for a JSON answer,
 * classifyStreamAnswer for a stream — so this function only owns the checks both wires share.
 */
async function readUpstreamAnswer(ctx: RequestContext, model: string, upstream: Response, startedAt: number): Promise<Attempt> {
  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '');
    return {
      kind: 'refused',
      status: upstream.status,
      code: errorCodeOf(text),
      latencyMs: Date.now() - startedAt,
      retryAfter: upstream.headers.get('retry-after'),
    };
  }
  if (upstream.body === null) return { kind: 'broken', reason: 'no body' };
  const text = await upstream.text().catch(() => '');
  if (text === '') return { kind: 'broken', reason: 'empty body' };
  return ctx.stream ? classifyStreamAnswer(ctx, model, text) : classifyJsonAnswer(ctx, model, text);
}

/**
 * Answer one client request from one dialled answer: translate JSON, or relay the buffered stream
 * through the wire translators. The failures header rides on the answer when refusals were recovered from.
 */
async function deliver(
  answer: DialledAnswer,
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  failures: TrailRow[],
  res: ServerResponse,
): Promise<void> {
  const headers = trailHeader(failures);
  if (!stream) {
    if (answer.json === null) throw new Error('a JSON answer holds no JSON');
    if (clientApi === 'openai') sendJson(res, 200, openAIJsonWithModel(answer.json, answer.model), headers);
    else sendJson(res, 200, openAIToAnthropic(answer.json, answer.model), headers);
    return;
  }
  if (answer.streamText === null) throw new Error('a stream answer holds no stream');
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
  try {
    if (clientApi === 'openai') {
      for await (const frame of openAIStreamWithModel(streamOf(answer.streamText), answer.model)) res.write(frame);
    } else {
      for await (const event of anthropicEventsFromOpenAI(streamOf(answer.streamText), answer.model)) res.write(event);
    }
  } finally {
    res.end();
  }
}

/** What a refusal means for the model now being tried: retry it, move to the next model, or stop dialling. */
type RefusalVerdict = 'retry' | 'next' | 'stop';

/** The family one refusal belongs to, by status alone. A 504 here is the entry's own code for a
 * timeout, an unreachable gateway, or a deadline already spent — it retries like a 5xx. */
type RefusalKind = 'auth' | 'bad-request' | 'retryable';

function refusalKindOf(status: number): RefusalKind {
  if (status === 401) return 'auth';
  if (status === 400) return 'bad-request';
  return 'retryable';
}

/**
 * REFUSAL DECISION TABLE — one refusal, by family:
 *
 *   status    | kind        | trail | model state  | dialling after
 *   ----------|-------------|-------|--------------|----------------------------------------
 *   401       | auth        | yes   | —            | stop: the key is missing or refused
 *   400       | bad-request | yes   | failure + 1  | next model; a 2nd in a row stops the prompt
 *   429 / 5xx | retryable   | yes   | failure + 1  | retry, unless 3 models refused alike → stop
 *   504 *     | retryable   | yes   | failure + 1  | retry (timeout / unreachable / deadline)
 *
 * Classification (refusalKindOf) is separate from the verdict and its side effects below, so the
 * table above is the one place that says what each status means.
 */
function recordRefusal(ctx: RequestContext, attempt: Extract<Attempt, { kind: 'refused' }>, model: string): RefusalVerdict {
  ctx.failures.push({ status: attempt.status, code: attempt.code, model, latency_ms: attempt.latencyMs });
  const kind = refusalKindOf(attempt.status);
  if (kind === 'auth') {
    keyDead = true;
    log('the gateway answered 401: the entrant key is missing or refused');
    return 'stop';
  }
  if (kind === 'bad-request') {
    noteFailure(model);
    log(`call to ${model} answered HTTP 400${attempt.code === '' ? '' : ` (${attempt.code})`}`);
    ctx.badRequests += 1;
    return ctx.badRequests >= 2 ? 'stop' : 'next';
  }
  noteFailure(model);
  log(`call to ${model} answered HTTP ${attempt.status}${attempt.code === '' ? '' : ` (${attempt.code})`} after ${attempt.latencyMs} ms`);
  if (systemicRefusal(attempt.status, attempt.code, model)) {
    gatewayDead = true;
    log(`the gateway refused HTTP ${attempt.status} (${attempt.code}) on three models: stopping dialling this trial`);
    return 'stop';
  }
  return 'retry';
}

/** What trying one model for one request came to: it served, it is spent, or dialling stops. */
type ModelOutcome = 'served' | 'next' | 'stop';

/**
 * The pause before the next retry of a refused call, in milliseconds, or null when no retry
 * fits: the retry budget for this model is spent, or the wait would run past the request's
 * deadline (a second of margin is kept so the retried call itself has time to answer).
 */
function nextRetryDelay(ctx: RequestContext, retries: number, retryAfter: string | null): number | null {
  if (retries >= MAX_MODEL_RETRIES) return null;
  const delay = retryDelayMs(retries, retryAfter);
  if (Date.now() >= ctx.deadline - delay - 1000) return null;
  return delay;
}

/** What a dialled answer means for this model: serve it, try the model once more after a token
 * miss, or move on. A token miss is not a failure of the model — it answered well, the prompt's
 * token is just not in it — so its failure streak is untouched and the miss is kept as a fallback
 * instead. */
type AnswerOutcome = 'served' | 'retry-token' | 'next';

async function handleAnswer(
  ctx: RequestContext,
  answer: DialledAnswer,
  tokenMissRetried: boolean,
  res: ServerResponse,
): Promise<AnswerOutcome> {
  noteSuccess(answer.model);
  if (answer.tokenOk) {
    await deliver(answer, ctx.clientApi, ctx.stream, ctx.failures, res);
    return 'served';
  }
  ctx.fallbacks.push(answer);
  log(`call to ${answer.model} answered without the token`);
  return tokenMissRetried ? 'next' : 'retry-token';
}

/**
 * Serve one request from one model: dial it, retry a refusal on it, retry a token miss once,
 * and say what came of it.
 */
async function serveFromModel(ctx: RequestContext, model: string, res: ServerResponse): Promise<ModelOutcome> {
  let retries = 0;
  /** Whether this model has had its one retry after a token miss, for this prompt. */
  let tokenMissRetried = false;
  for (;;) {
    const attempt = await tryModel(ctx, model);
    if (attempt.kind === 'answer') {
      const outcome = await handleAnswer(ctx, attempt.answer, tokenMissRetried, res);
      if (outcome === 'served') return 'served';
      if (outcome === 'retry-token') {
        tokenMissRetried = true;
        continue;
      }
      return 'next';
    }
    if (attempt.kind === 'broken') {
      // A 2xx whose body is unusable: the model is no good for this prompt, and no other model may
      // be blamed for it, so move on without a trail row.
      noteFailure(model);
      log(`call to ${model} answered an unusable body (${attempt.reason})`);
      return 'next';
    }
    const verdict = recordRefusal(ctx, attempt, model);
    if (verdict === 'stop') return 'stop';
    if (verdict === 'next') return 'next';
    const delay = nextRetryDelay(ctx, retries, attempt.retryAfter);
    if (delay === null) return 'next';
    retries += 1;
    await pause(delay);
    continue;
  }
}

/**
 * The first usable answer that carries the token under the normalized comparison (another
 * spelling or case), or undefined when none does. The normalized re-check is the fallback's
 * second chance: an answer that names the token in another spelling still names it.
 */
function normalizedFallback(fallbacks: DialledAnswer[], token: string): DialledAnswer | undefined {
  return fallbacks.find((answer) => normalizedCarries(answerTextOf(answer), token));
}

/**
 * The best usable answer when no model carried the token: the first that carries it under the
 * normalized comparison, else the best model's answer, whose wire checks still count even
 * though the prompt itself earns nothing. With no token to carry, the best model's answer is
 * the fallback by definition.
 */
function pickFallback(fallbacks: DialledAnswer[], token: string | null): DialledAnswer {
  if (token === null) return fallbacks[0];
  return normalizedFallback(fallbacks, token) ?? fallbacks[0];
}

/**
 * Serve the best usable answer when no model carried the token, or false when no usable answer
 * was kept. This is the request's last resort, after every candidate has been tried.
 */
async function serveFallback(ctx: RequestContext, res: ServerResponse): Promise<boolean> {
  if (ctx.fallbacks.length === 0) return false;
  await deliver(pickFallback(ctx.fallbacks, ctx.token), ctx.clientApi, ctx.stream, ctx.failures, res);
  return true;
}

/** Everything one client request carries through routing and failover, built from its body. */
function contextOf(clientApi: 'openai' | 'anthropic', body: Json, startedAt: number): RequestContext {
  return {
    clientApi,
    body,
    stream: body.stream === true,
    token: tokenOf(body),
    deadline: startedAt + REQUEST_DEADLINE_MS,
    failures: [],
    fallbacks: [],
    badRequests: 0,
  };
}

/** What routing one request through its candidates came to: it was served, dialling stopped, or
 * every candidate was tried (or the deadline / a dead key or gateway cut the walk short). */
type RouteOutcome = 'served' | 'stop' | 'exhausted';

async function serveCandidates(ctx: RequestContext, candidates: ModelRow[], res: ServerResponse): Promise<RouteOutcome> {
  for (const candidate of candidates) {
    if (keyDead || gatewayDead || Date.now() > ctx.deadline - 1000) return 'exhausted';
    const outcome = await serveFromModel(ctx, candidate.id, res);
    if (outcome === 'served') return 'served';
    if (outcome === 'stop') return 'stop';
  }
  return 'exhausted';
}

/** POST /v1/chat/completions or POST /v1/messages: route by rating, fail over, answer in the client's wire. */
async function serveRequest(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const startedAt = Date.now();
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const ctx = contextOf(clientApi, body, startedAt);
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0 || keyDead || gatewayDead) {
    return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'no free model can serve this request');
  }
  if ((await serveCandidates(ctx, candidates, res)) === 'served') return;
  if (await serveFallback(ctx, res)) return;
  if (!res.headersSent) {
    sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'every free model that meets this request refused it', trailHeader(ctx.failures));
  }
}

/** The POST routes this server answers, by method and path, with the wire each speaks. */
const ROUTES: Record<string, 'openai' | 'anthropic'> = {
  'POST /v1/chat/completions': 'openai',
  'POST /v1/messages': 'anthropic',
};

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  const wire = ROUTES[`${req.method ?? ''} ${url.pathname}`];
  if (wire !== undefined) {
    await serveRequest(wire, req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count.
 * Retried with exponential backoff: the env may still be starting when the entry first dials it,
 * and a register the env refused (ok: false) is a failure, not a registration. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && reply.ok === false) throw new Error('POST /register was refused by the env');
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = err as Error;
      if (attempt < 4) await pause(200 * 2 ** attempt);
    }
  }
  throw lastError ?? new Error('POST /register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
