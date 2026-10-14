import { pairScores, streamGenerate, type PairScores } from "../funcs";
import { type Doc, wordCount } from "./dataset";

export type Head = "gru" | "diffalign";
export type Signal = "same" | "different";

const LANG_NAME: Record<string, string> = { de: "German", fr: "French", it: "Italian" };

export function runHead(doc: Doc, head: Head): PairScores | null {
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

export async function runNote(
  doc: Doc,
  signal: Signal,
  onChunk: (text: string) => void,
  isCancelled: () => boolean,
  onProgress?: (fraction: number) => void,
): Promise<string> {
  const maxTokens = 200;
  const result = await streamGenerate({
    prompt: promptFor(doc, signal),
    system: systemFor(doc.lang, signal),
    maxTokens,
    temperature: signal === "same" ? 0 : 0.2,
    onChunk,
    isCancelled,
    onProgress: onProgress ? (generated, max) => onProgress(max > 0 ? generated / max : 0) : undefined,
  });
  return result.text;
}
