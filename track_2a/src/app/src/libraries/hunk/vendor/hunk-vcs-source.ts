// Local shim for the `@hunk/vcs/source` workspace package the vendored Hunk
// source imports (see core/changeset/fileSource.ts). Mirrors the upstream
// contract in packages/hunk-vcs/src/source.ts so the call sites narrow
// correctly, without pulling in the rest of the Hunk workspace.
import { readFile } from "node:fs/promises";

/** Default byte ceiling for exact source text loaded for expanded context. */
export const DEFAULT_SOURCE_TEXT_MAX_BYTES = 1_000_000;

/** Structural result shared by bounded filesystem source readers. */
export type LimitedSourceTextResult = string | null | { kind: "too-large"; maxBytes: number };

/** Read one filesystem source as text without exceeding the supplied byte ceiling. */
export async function readFileTextWithLimit(
  absolutePath: string,
  maxBytes: number,
): Promise<LimitedSourceTextResult> {
  try {
    const buffer = await readFile(absolutePath);
    if (buffer.byteLength > maxBytes) {
      return { kind: "too-large", maxBytes };
    }
    return buffer.toString("utf8");
  } catch {
    return null;
  }
}
