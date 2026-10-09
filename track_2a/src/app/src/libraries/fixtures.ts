// TODO -> Remove this dummy data and add actual data
import { diffTexts } from "./facade";
import type { HunkDiffFileInput } from "./hunk/opentui";

const BEFORE = `export interface DiffRequest {
  left: string;
  right: string;
}

export function summarize(request: DiffRequest): string {
  const total = request.left.length + request.right.length;
  return \`\${total} characters compared\`;
}

export const DEFAULT_LAYOUT = "split";
`;

const AFTER = `export interface DiffRequest {
  left: string;
  right: string;
  language?: string;
}

export function summarize(request: DiffRequest): string {
  const total = request.left.length + request.right.length;
  const label = request.language ?? "text";
  return \`\${total} characters compared as \${label}\`;
}

export const DEFAULT_LAYOUT = "unified";
`;

export const dummyDiff: HunkDiffFileInput = diffTexts(BEFORE, AFTER, {
  language: "typescript",
  path: "src/review-summary.ts",
  id: "review-summary",
  note: "Threads the language through summarize() and flips the default layout to unified.",
});

export const dummyFiles: HunkDiffFileInput[] = [
  diffTexts(BEFORE, AFTER, {
    language: "typescript",
    path: "src/review-summary.ts",
    id: "review-summary",
  }),
  diffTexts("First line.\nSecond line.\n", "First line.\nSecond line changed.\n", {
    language: "markdown",
    path: "notes/summary.md",
    id: "notes-summary",
  }),
  diffTexts("const a = 1;\n", "const a = 2;\n", {
    language: "typescript",
    path: "src/semantic/diff.ts",
    id: "semantic-diff",
  }),
];
