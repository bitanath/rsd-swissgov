export type HeadId = "gru" | "diffalign" | "combined";
export type Signal = "same" | "different";
export type Phase = "loading" | "capturing" | "classifying" | "writing";

export interface AnalyseRequest {
  type: "analyse";
  id: number;
  textA: string;
  textB: string;
  wcA: number;
  wcB: number;
  lang: string;
}

export interface CancelRequest {
  type: "cancel";
  id: number;
}

export type WorkerRequest = AnalyseRequest | CancelRequest;

export interface PhaseEvent {
  type: "phase";
  id: number;
  phase: Phase;
  pass?: number;
}
export interface ProgressEvent {
  type: "progress";
  id: number;
  phase: Phase;
  fraction: number;
  pass?: number;
}
export interface CombinedScores {
  a: Float32Array;
  b: Float32Array;
  //FIXME: bitmask per word: 1 = GRU flagged it, 2 = DiffAlign flagged it, 3 = both.
  provA: Uint8Array;
  provB: Uint8Array;
}
export interface ScoresEvent {
  type: "scores";
  id: number;
  a: Float32Array;
  b: Float32Array;
  signal: Signal;
  combined?: CombinedScores;
}
export interface NoteEvent {
  type: "note";
  id: number;
  text: string;
}
export interface NoteProgressEvent {
  type: "noteProgress";
  id: number;
  fraction: number;
}
export interface DoneEvent {
  type: "done";
  id: number;
  note: string;
}
export interface ErrorEvent {
  type: "error";
  id: number;
  message: string;
}
export interface CancelledEvent {
  type: "cancelled";
  id: number;
}

export type WorkerEvent =
  | PhaseEvent
  | ProgressEvent
  | ScoresEvent
  | NoteEvent
  | NoteProgressEvent
  | DoneEvent
  | ErrorEvent
  | CancelledEvent;
