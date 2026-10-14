import { ensureModel } from "../funcs";
import { runNote, signalFromScores } from "../libraries/analyze";
import type { Doc } from "../libraries/dataset";
import { pairScoresProgress } from "./ffi";
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

    const scores = pairScoresProgress(
      req.textA,
      req.textB,
      req.head,
      req.wcA,
      req.wcB,
      (phase, done, total) => {
        const fraction = total > 0 ? done / total : 0;
        if (phase === 2) post({ type: "progress", id, phase: "classifying", fraction });
        else post({ type: "progress", id, phase: "capturing", pass: phase + 1, fraction });
      },
    );
    if (cancelled.has(id)) return post({ type: "cancelled", id });
    if (!scores) return post({ type: "error", id, message: "pair_scores_ex failed" });

    const signal = signalFromScores(scores);
    post({ type: "scores", id, a: scores.a, b: scores.b, signal });

    post({ type: "phase", id, phase: "writing" });
    const doc = { text_a: req.textA, text_b: req.textB, lang: req.lang } as unknown as Doc;
    const text = await runNote(
      doc,
      signal,
      (t) => post({ type: "note", id, text: t }),
      () => cancelled.has(id),
      (fraction) => post({ type: "noteProgress", id, fraction }),
    );
    if (cancelled.has(id)) return post({ type: "cancelled", id });
    post({ type: "done", id, note: text });
  } catch (err) {
    post({ type: "error", id, message: err instanceof Error ? err.message : String(err) });
  } finally {
    cancelled.delete(id);
  }
}
