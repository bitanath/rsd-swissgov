import { dlopen, JSCallback, ptr } from "bun:ffi";

import { ensureModel, LIB_PATH } from "../funcs";
import type { HeadId } from "./protocol";

export interface PairScores {
  a: Float32Array;
  b: Float32Array;
}

export type HeadProgress = (phase: number, done: number, total: number) => void;

const { symbols } = dlopen(LIB_PATH, {
  pair_scores_ex: {
    args: ["ptr", "cstring", "cstring", "i32", "ptr", "usize", "ptr", "usize", "ptr"],
    returns: "i32",
  },
});

const enc = new TextEncoder();
function encodeString(s: string): Uint8Array {
  const bytes = enc.encode(s);
  const out = new Uint8Array(bytes.length + 1);
  out.set(bytes, 0);
  return out;
}

export function pairScoresProgress(
  textA: string,
  textB: string,
  head: HeadId,
  wcA: number,
  wcB: number,
  onProgress?: HeadProgress,
): PairScores | null {
  const handle = ensureModel();
  const outA = new Float32Array(wcA);
  const outB = new Float32Array(wcB);
  const za = encodeString(textA);
  const zb = encodeString(textB);
  const cb = new JSCallback(
    (phase: number, done: number, total: number) => onProgress?.(phase, done, total),
    { args: ["u32", "u32", "u32"], returns: "void" },
  );
  try {
    const rc = symbols.pair_scores_ex(
      handle,
      za,
      zb,
      head === "diffalign" ? 1 : 0,
      ptr(outA),
      wcA,
      ptr(outB),
      wcB,
      cb.ptr,
    ) as number;
    if (rc !== 0) return null;
    return { a: outA, b: outB };
  } finally {
    cb.close();
  }
}
