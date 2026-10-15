import { ensureModel } from "../../funcs";
import { combinedFrom, runNoteCombined } from "../clit/analyze";
import type { Doc } from "../clit/dataset";
import { pairScoresBothProgress } from "./ffi";
import type { AnalyseRequest, WorkerEvent, WorkerRequest } from "./protocol";

interface WorkerScope {
  postMessage: (event: WorkerEvent) => void;
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
}

const ctx = self as unknown as WorkerScope;

const cancelled = new Set<number>();
const post = (event: WorkerEvent): void => ctx.postMessage(event);

let chain: Promise<void> = Promise.resolve();

ctx.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const req = event.data;
  if (req.type === "cancel") {
    cancelled.add(req.id);
    return;
  }
  chain = chain.then(() => run(req));
};

async function run(req: AnalyseRequest): Promise<void> {
  const { id } = req;
  try {
    post({ type: "phase", id, phase: "loading" });
    ensureModel();
    if (cancelled.has(id)) return post({ type: "cancelled", id });

    //TODO: one capture, both heads -> ^r/^k/^d are all instant.
    const both = pairScoresBothProgress(req.textA, req.textB, req.wcA, req.wcB, (phase, done, total) => {
      const fraction = total > 0 ? done / total : 0;
      if (phase === 2) post({ type: "progress", id, phase: "classifying", fraction });
      else post({ type: "progress", id, phase: "capturing", pass: phase + 1, fraction });
    });
    if (cancelled.has(id)) return post({ type: "cancelled", id });
    if (!both) return post({ type: "error", id, message: "pair_scores_both_ex failed" });

    //TODO: The agent runs once per pair, on the union (combined) labels; every mode reuses it.
    const { combined, signal } = combinedFrom(both.gru, both.diff, req.lang);
    post({ type: "scores", id, a: combined.a, b: combined.b, signal, combined });

    post({ type: "phase", id, phase: "writing" });
    const doc = { text_a: req.textA, text_b: req.textB, lang: req.lang } as unknown as Doc;
    const onChunk = (t: string) => post({ type: "note", id, text: t });
    const isCancelled = () => cancelled.has(id);
    const onNoteProgress = (fraction: number) => post({ type: "noteProgress", id, fraction });
    const text = await runNoteCombined(doc, combined, signal, onChunk, isCancelled, onNoteProgress);
    if (cancelled.has(id)) return post({ type: "cancelled", id });
    post({ type: "done", id, note: text });
  } catch (err) {
    post({ type: "error", id, message: err instanceof Error ? err.message : String(err) });
  } finally {
    cancelled.delete(id);
  }
}
