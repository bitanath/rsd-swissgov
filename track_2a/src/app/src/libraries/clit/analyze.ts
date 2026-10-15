import { pairScores, streamGenerate, type PairScores } from "../../funcs";
import { words, type Doc, wordCount } from "./dataset";

export type Head = "gru" | "diffalign" | "combined";
export type Signal = "same" | "different";

export const HEAD_LABEL: Record<Head, string> = {
  gru: "pairprobe (supervised)",
  diffalign: "diffalign (unsupervised)",
  combined: "combined (either head)",
};

export const T_GRU: Record<string, number> = { de: 0.2, fr: 0.16, it: 0.18 };
export const T_DIFF: Record<string, number> = { de: 0.27, fr: 0.3, it: 0.27 };
export const PROV_GRU = 1;
export const PROV_DIFF = 2;

export function headMask(head: Head | null): number {
  if (head === "gru") return PROV_GRU;
  if (head === "diffalign") return PROV_DIFF;
  return PROV_GRU | PROV_DIFF;
}

export interface Combined {
  a: Float32Array;
  b: Float32Array;
  provA: Uint8Array;
  provB: Uint8Array;
}

const LANG_NAME: Record<string, string> = { de: "German", fr: "French", it: "Italian" };

export function runHead(doc: Doc, head: "gru" | "diffalign"): PairScores | null {
  return pairScores(doc.text_a, doc.text_b, head, wordCount(doc.text_a), wordCount(doc.text_b));
}

export function signalFromScores(scores: PairScores | null): Signal {
  if (!scores) return "different";
  let max = 0;
  for (const v of scores.a) if (v > max) max = v;
  for (const v of scores.b) if (v > max) max = v;
  return max > 0.5 ? "different" : "same";
}

export function signalFromGold(doc: Doc): Signal {
  const differs = doc.labels_a.some((v) => v > 0.5) || doc.labels_b.some((v) => v > 0.5);
  return differs ? "different" : "same";
}

function provAt(g: number, d: number, tg: number, td: number): number {
  let p = 0;
  if (g > tg) p |= PROV_GRU;
  if (d > td) p |= PROV_DIFF;
  return p;
}

export function combinedFrom(
  gru: PairScores,
  diff: PairScores,
  lang: string,
): { combined: Combined; signal: Signal } {
  const tg = T_GRU[lang] ?? 0.1;
  const td = T_DIFF[lang] ?? 0.22;
  const n = Math.min(gru.a.length, diff.a.length);
  const m = Math.min(gru.b.length, diff.b.length);
  const provA = new Uint8Array(n);
  const provB = new Uint8Array(m);
  const a = new Float32Array(n);
  const b = new Float32Array(m);
  let flagged = 0;
  for (let i = 0; i < n; i++) {
    const g = gru.a[i] ?? 0;
    const d = diff.a[i] ?? 0;
    const p = provAt(g, d, tg, td);
    provA[i] = p;
    a[i] = Math.max(g, d);
    if (p) flagged++;
  }
  for (let i = 0; i < m; i++) {
    const g = gru.b[i] ?? 0;
    const d = diff.b[i] ?? 0;
    const p = provAt(g, d, tg, td);
    provB[i] = p;
    b[i] = Math.max(g, d);
    if (p) flagged++;
  }
  return { combined: { a, b, provA, provB }, signal: flagged > 0 ? "different" : "same" };
}

export function markWords(text: string, flags: Uint8Array): string {
  return words(text)
    .map((w, i) => (flags[i] ? `***${w}***` : w))
    .join(" ");
}


const BASE =
  "You are a strict bilingual document evaluator. You judge multilingual documents for similarity or difference. You ALWAYS answer in English only — " +
  "never in German, French or Italian, no matter what language the documents are in.";

function systemFor(lang: string, signal: Signal): string {
  const name = LANG_NAME[lang] ?? lang.toUpperCase();
  if (signal === "same") {
    return (
      `${BASE} An English document and its ${name} counterpart are semantically the SAME. ` +
      `Reply with exactly this sentence and nothing else: The docs are semantically the same`
    );
  }
  return (
    `${BASE} An English document and its ${name} counterpart are semantically DIFFERENT. ` +
    `In English, give the reason for the semantic difference in 1-2 sentences, beginning ` +
    `with "The docs are semantically different:".`
  );
}

export function promptFor(doc: Doc, signal: Signal): string {
  const name = LANG_NAME[doc.lang] ?? doc.lang.toUpperCase();
  const tail =
    signal === "same"
      ? "Verdict: SAME. Answer in English with exactly: The docs are semantically the same"
      : 'Verdict: DIFFERENT. In English, explain the semantic difference in 1-2 sentences, starting with "The docs are semantically different:".';
  return `English document:\n${doc.text_a}\n\n${name} document:\n${doc.text_b}\n\n${tail}`;
}

const MARK_HELP =
  "The documents are given in Markdown. Spans wrapped in ***bold italic*** were flagged by the diff models as differing — use them as evidence.";

function systemForCombined(lang: string, signal: Signal): string {
  const name = LANG_NAME[lang] ?? lang.toUpperCase();
  if (signal === "same") {
    return (
      `${BASE} ${MARK_HELP} An English document and its ${name} counterpart are semantically the SAME. ` +
      `Reply with exactly this sentence and nothing else: The docs are semantically the same`
    );
  }
  return (
    `${BASE} ${MARK_HELP} An English document and its ${name} counterpart are semantically DIFFERENT. ` +
    `Using the flagged spans as evidence, give the reason for the semantic difference in 1-2 English sentences, beginning ` +
    `with "The docs are semantically different:".`
  );
}

export function promptForCombined(doc: Doc, flagsA: Uint8Array, flagsB: Uint8Array, signal: Signal): string {
  const name = LANG_NAME[doc.lang] ?? doc.lang.toUpperCase();
  const tail =
    signal === "same"
      ? "Verdict: SAME. Answer in English with exactly: The docs are semantically the same"
      : 'Verdict: DIFFERENT. Using the ***flagged*** spans, explain the semantic difference in 1-2 sentences, starting with "The docs are semantically different:".';
  return `English document (Markdown):\n${markWords(doc.text_a, flagsA)}\n\n${name} document (Markdown):\n${markWords(doc.text_b, flagsB)}\n\n${tail}`;
}

async function streamNote(
  prompt: string,
  system: string,
  signal: Signal,
  onChunk: (text: string) => void,
  isCancelled: () => boolean,
  onProgress?: (fraction: number) => void,
): Promise<string> {
  const maxTokens = 200;
  const result = await streamGenerate({
    prompt,
    system,
    maxTokens,
    temperature: signal === "same" ? 0 : 0.2,
    onChunk,
    isCancelled,
    onProgress: onProgress ? (generated, max) => onProgress(max > 0 ? generated / max : 0) : undefined,
  });
  return result.text;
}

export async function runNote(
  doc: Doc,
  signal: Signal,
  onChunk: (text: string) => void,
  isCancelled: () => boolean,
  onProgress?: (fraction: number) => void,
): Promise<string> {
  return streamNote(promptFor(doc, signal), systemFor(doc.lang, signal), signal, onChunk, isCancelled, onProgress);
}

//TODO: grounded note for combined mode (marked union differences).
export async function runNoteCombined(
  doc: Doc,
  combined: Combined,
  signal: Signal,
  onChunk: (text: string) => void,
  isCancelled: () => boolean,
  onProgress?: (fraction: number) => void,
): Promise<string> {
  return streamNote(
    promptForCombined(doc, combined.provA, combined.provB, signal),
    systemForCombined(doc.lang, signal),
    signal,
    onChunk,
    isCancelled,
    onProgress,
  );
}
