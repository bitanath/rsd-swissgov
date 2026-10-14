import "./devs";

import { CString, dlopen, ptr, type Pointer } from "bun:ffi";
import libPath from "../libelthur.bin" with { type: "file" };

// NOTE: Built FFI surface we leverage here. The library is embedded in this Bun binary at build time (see scripts/build.sh) and loaded through bun:ffi.
const { symbols } = dlopen(libPath, {
  model_create_ex: { args: ["cstring", "i32", "f64", "f64", "i32"], returns: "ptr" },
  model_set_threads: { args: ["ptr", "u32"], returns: "void" },
  model_free: { args: ["ptr"], returns: "void" },
  stream_start_model: {
    args: ["ptr", "ptr", "usize", "cstring", "i32", "f64", "f64", "i32"],
    returns: "ptr",
  },
  stream_poll: { args: ["ptr", "ptr", "usize"], returns: "i64" }, // >0 bytes, 0 idle, -1 done, -2 error.
  stream_free: { args: ["ptr"], returns: "void" },
  stream_hit_length_limit: { args: ["ptr"], returns: "bool" },
  stream_generated_tokens: { args: ["ptr"], returns: "usize" },
  stream_prompt_tokens: { args: ["ptr"], returns: "usize" },
  stream_ttft_nanoseconds: { args: ["ptr"], returns: "u64" },
  free_string: { args: ["ptr"], returns: "void" },
  pair_scores: {
    args: ["ptr", "cstring", "cstring", "i32", "ptr", "usize", "ptr", "usize"],
    returns: "i32",
  },
});

export const LIB_PATH = libPath;

export const DEFAULT_SYSTEM = "You are Apertus, a helpful assistant.";
export const CHAT_MAX_TOKENS = 1024;
const DEFAULT_MAX_TOKENS = 256;
const DEFAULT_TEMPERATURE = 0.7;
const DEFAULT_TOP_P = 0.9;
const DEFAULT_TOP_K = 40;

export interface ChatMessage {
  role: string;
  content: string;
}

export interface GenerateOptions {
  prompt: string;
  system?: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
}

export interface StreamOptions extends GenerateOptions {
  onChunk: (text: string) => void;
  isCancelled?: () => boolean;
  onProgress?: (generated: number, max: number) => void;
}

export interface StreamResult {
  text: string;
  truncated: boolean;
  tokens: number;
  promptTokens: number;
  ttftMs: number;
}

export class EngineFailed extends Error {
  constructor() {
    super("elthur returned NULL");
    this.name = "EngineFailed";
  }
}

export class EngineCancelled extends Error {
  constructor() {
    super("elthur generation cancelled");
    this.name = "EngineCancelled";
  }
}

const enc = new TextEncoder();

function encodeString(s: string): Uint8Array {
  const b = enc.encode(s);
  const out = new Uint8Array(b.length + 1);
  out.set(b, 0);
  out[out.length - 1] = 0;
  return out;
}

function paramsOf(opts: GenerateOptions) {
  return {
    maxTokens: opts.maxTokens ?? DEFAULT_MAX_TOKENS,
    temperature: opts.temperature ?? DEFAULT_TEMPERATURE,
    topP: opts.topP ?? DEFAULT_TOP_P,
    topK: opts.topK ?? DEFAULT_TOP_K,
  };
}

let model: Pointer | null = null;

export function ensureModel(): Pointer {
  if (model === null) {
    const systemZ = encodeString(DEFAULT_SYSTEM);
    model = symbols.model_create_ex(systemZ, CHAT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K) as Pointer | null;
    void systemZ;
    if (model === null) throw new EngineFailed();
  }
  return model;
}

function beginStream(messages: ChatMessage[], system: string, opts: GenerateOptions): Pointer {
  const m = ensureModel();
  const zipped = messages.map((msg) => ({ role: encodeString(msg.role), content: encodeString(msg.content) }));
  const msgs = new BigUint64Array(messages.length * 2);
  for (let i = 0; i < messages.length; i++) {
    const p = zipped[i]!;
    msgs[i * 2] = BigInt(ptr(p.role));
    msgs[i * 2 + 1] = BigInt(ptr(p.content));
  }
  const systemZ = encodeString(system);
  const p = paramsOf(opts);
  const handle = symbols.stream_start_model(m, msgs, messages.length, systemZ, p.maxTokens, p.temperature, p.topP, p.topK) as Pointer | null;
  void zipped;
  void msgs;
  void systemZ;
  if (handle === null) throw new EngineFailed();
  return handle;
}

export function pollStream(handle: Pointer, out: Uint8Array): number {
  return Number(symbols.stream_poll(handle, out, out.length)); //NOTE : - Low-level poll. Exposed for callers that want to drive the loop themselves.
}

export function freeStream(handle: Pointer): void {
  symbols.stream_free(handle);
}

export function streamTruncated(handle: Pointer): boolean {
  return symbols.stream_hit_length_limit(handle);
}

export function startStream(opts: GenerateOptions): Pointer {
  return beginStream([{ role: "user", content: opts.prompt }], opts.system ?? DEFAULT_SYSTEM, opts);
}

export function streamTimings(handle: Pointer): { generated: number; promptTokens: number; ttftMs: number } {
  return {
    generated: Number(symbols.stream_generated_tokens(handle)),
    promptTokens: Number(symbols.stream_prompt_tokens(handle)),
    ttftMs: Number(symbols.stream_ttft_nanoseconds(handle)) / 1e6, //TODO for debugging etc (borderline usable is >10 tok/s and >50 prefill tok/s )
  };
}

export type PairHead = "gru" | "diffalign";

export interface PairScores {
  a: Float32Array;
  b: Float32Array;
}

/**
 * Run one RSD head over a single document pair. The heavy lifting (L14-17
 * capture, word pooling, GRU/DiffAlign) happens in the Zig core; this just hands
 * over the two texts and receives one f32 per whitespace word per side.
 */
export function pairScores(
  textA: string,
  textB: string,
  head: PairHead,
  wordCountA: number,
  wordCountB: number,
): PairScores | null {
  const handle = ensureModel();
  const outA = new Float32Array(wordCountA);
  const outB = new Float32Array(wordCountB);
  const za = encodeString(textA);
  const zb = encodeString(textB);
  const rc = symbols.pair_scores(
    handle,
    za,
    zb,
    head === "diffalign" ? 1 : 0,
    ptr(outA),
    wordCountA,
    ptr(outB),
    wordCountB,
  ) as number;
  void za;
  void zb;
  if (rc !== 0) return null;
  return { a: outA, b: outB };
}

export function freeModel(): void {
  if (model !== null) {
    symbols.model_free(model);
    model = null;
  }
}

export function generate(opts: GenerateOptions): StreamResult {
  const handle = beginStream([{ role: "user", content: opts.prompt }], opts.system ?? DEFAULT_SYSTEM, opts);
  const out = new Uint8Array(65536);
  const decoder = new TextDecoder();
  let text = "";
  try {
    for (;;) {
      const n = Number(symbols.stream_poll(handle, out, out.length));
      if (n === -2) throw new EngineFailed();
      if (n > 0) text += decoder.decode(out.subarray(0, n), { stream: true });
      else if (n === -1) break;
      else Bun.sleepSync(5);
    }
    text += decoder.decode();
    const t = streamTimings(handle);
    return { text, truncated: symbols.stream_hit_length_limit(handle), tokens: t.generated, promptTokens: t.promptTokens, ttftMs: t.ttftMs };
  } finally {
    symbols.stream_free(handle);
  }
}

export async function streamGenerate(opts: StreamOptions): Promise<StreamResult> {
  const handle = beginStream([{ role: "user", content: opts.prompt }], opts.system ?? DEFAULT_SYSTEM, opts);
  const isCancelled = opts.isCancelled ?? (() => false);
  const out = new Uint8Array(65536);
  const decoder = new TextDecoder();
  let text = "";
  try {
    for (;;) {
      if (isCancelled()) throw new EngineCancelled();
      const n = Number(symbols.stream_poll(handle, out, out.length));
      if (n === -2) throw new EngineFailed();
      if (n > 0) {
        text += decoder.decode(out.subarray(0, n), { stream: true });
        opts.onChunk(text);
        opts.onProgress?.(Number(symbols.stream_generated_tokens(handle)), opts.maxTokens ?? DEFAULT_MAX_TOKENS);
      } else if (n === -1) {
        break;
      } else {
        await Bun.sleep(15);
      }
    }
    text += decoder.decode();
    opts.onChunk(text);
    const t = streamTimings(handle);
    return { text, truncated: symbols.stream_hit_length_limit(handle), tokens: t.generated, promptTokens: t.promptTokens, ttftMs: t.ttftMs };
  } finally {
    symbols.stream_free(handle);
  }
}
