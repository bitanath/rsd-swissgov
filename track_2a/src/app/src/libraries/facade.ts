import { readFileSync } from "node:fs";
import { parseDiffFromFile } from "@pierre/diffs";

import { createHunkDiffFile, type HunkDiffFileInput } from "./hunk/opentui";

export interface DiffOptions {
  language?: string; //FIXME only DE, FR, IT and EN supported for now
  path?: string;
  id?: string;
  context?: number;
  note?: string; //per-file note that can be rendered in TUI
}

export function diffTexts(textA: string, textB: string, opts: DiffOptions = {}): HunkDiffFileInput {
  const path = opts.path ?? "text";
  const metadata = parseDiffFromFile(
    { cacheKey: `${path}:a`, contents: textA, name: path },
    { cacheKey: `${path}:b`, contents: textB, name: path },
    { context: opts.context ?? 3 },
    true,
  );
  return createHunkDiffFile({
    id: opts.id ?? "text",
    metadata,
    language: opts.language ?? "text",
    path,
    note: opts.note,
  }) 
}

export function diffFiles(leftPath: string, rightPath: string, opts: DiffOptions = {}): HunkDiffFileInput {
  const path = opts.path ?? leftPath;
  return diffTexts(readFileSync(leftPath, "utf8"), readFileSync(rightPath, "utf8"), { ...opts, path }) //HACK diff from text using the text in files
}
