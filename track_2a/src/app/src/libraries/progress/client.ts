//TODO: Analysis and progress bar handling through workers
import { words, type Doc } from "../clit/dataset";
import type { Combined } from "../clit/analyze";
import type { Phase, Signal, WorkerEvent, WorkerRequest } from "./protocol";

export interface AnalysisHandlers {
  onPhase?: (phase: Phase, pass?: number) => void;
  onProgress?: (phase: Phase, fraction: number, pass?: number) => void;
  onScores?: (a: Float32Array, b: Float32Array, signal: Signal, combined?: Combined) => void;
  onNote?: (text: string) => void;
  onNoteProgress?: (fraction: number) => void;
}

export interface AnalysisResult {
  combined: Combined | null;
  signal: Signal | null;
  note: string;
  cancelled: boolean;
  error: string | null;
}

export interface AnalysisClient {
  analyse(doc: Doc, handlers: AnalysisHandlers): Promise<AnalysisResult>;
  cancel(): void;
  dispose(): void;
}

interface Pending {
  handlers: AnalysisHandlers;
  combined: Combined | null;
  signal: Signal | null;
  note: string;
  resolve: (result: AnalysisResult) => void;
}

export function createAnalysisClient(): AnalysisClient {
  let worker: Worker | null = null;
  let nextId = 1;
  let inFlight = 0;
  const pending = new Map<number, Pending>();
  const inflight = new Map<string, Promise<AnalysisResult>>();

  const finish = (id: number, result: Omit<AnalysisResult, "combined" | "signal" | "note">): void => {
    const entry = pending.get(id);
    if (!entry) return;
    pending.delete(id);
    entry.resolve({ combined: entry.combined, signal: entry.signal, note: entry.note, ...result });
  };

  const route = (event: WorkerEvent): void => {
    const entry = pending.get(event.id);
    if (!entry) return;
    const { handlers } = entry;
    switch (event.type) {
      case "phase":
        handlers.onPhase?.(event.phase, event.pass);
        break;
      case "progress":
        handlers.onProgress?.(event.phase, event.fraction, event.pass);
        break;
      case "scores":
        entry.combined = event.combined ?? null;
        entry.signal = event.signal;
        handlers.onScores?.(event.a, event.b, event.signal, event.combined);
        break;
      case "note":
        entry.note = event.text;
        handlers.onNote?.(event.text);
        break;
      case "noteProgress":
        handlers.onNoteProgress?.(event.fraction);
        break;
      case "done":
        finish(event.id, { cancelled: false, error: null });
        break;
      case "cancelled":
        finish(event.id, { cancelled: true, error: null });
        break;
      case "error":
        finish(event.id, { cancelled: false, error: event.message });
        break;
    }
  };

  const ensureWorker = (): Worker => {
    if (!worker) {
      // Root-relative specifier matching the extra entrypoint passed to
      // `bun build --compile`; the URL form mis-resolves inside $bunfs here.
      worker = new Worker("./src/libraries/progress/worker.ts");
      worker.onmessage = (event: MessageEvent<WorkerEvent>) => route(event.data);
      worker.onerror = (event) => {
        const message = String((event as { message?: unknown }).message ?? "worker error");
        for (const id of [...pending.keys()]) finish(id, { cancelled: false, error: message });
      };
    }
    return worker;
  };

  const analyse: AnalysisClient["analyse"] = (doc, handlers) => {
    //FIXME: dedupe by doc so a head switch can never enqueue a second run.
    const existing = inflight.get(doc.id);
    if (existing) return existing;
    const active = ensureWorker();
    const id = nextId++;
    inFlight = id;
    const promise = new Promise<AnalysisResult>((resolve) => {
      pending.set(id, { handlers, combined: null, signal: null, note: "", resolve });
      const request: WorkerRequest = {
        type: "analyse",
        id,
        textA: doc.text_a,
        textB: doc.text_b,
        wcA: words(doc.text_a).length,
        wcB: words(doc.text_b).length,
        lang: doc.lang,
      };
      active.postMessage(request);
    });
    inflight.set(doc.id, promise);
    void promise.finally(() => {
      if (inflight.get(doc.id) === promise) inflight.delete(doc.id);
    });
    return promise;
  };

  const cancel = (): void => {
    if (worker && inFlight) worker.postMessage({ type: "cancel", id: inFlight } satisfies WorkerRequest);
  };

  const dispose = (): void => {
    worker?.terminate();
    worker = null;
    pending.clear();
    inflight.clear();
  };

  return { analyse, cancel, dispose };
}
