//TODO CLI entry point, Imma pass a --text-a or --text-b from the CLI to get a direct comparison without files
import { diffTexts } from "./facade";
import type { HunkDiffFileInput } from "../hunk/opentui";

function argValue(flag: string): string | undefined {
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

export function diffCliRequested(): boolean {
  return process.argv.includes("--texta") || process.argv.includes("--textb");
}

export function buildCliFile(): HunkDiffFileInput | undefined {
  if (!diffCliRequested()) {
    return undefined;
  }

  return diffTexts(argValue("--texta") ?? "", argValue("--textb") ?? "", {
    language: argValue("--language") ?? "text",
    path: argValue("--path") ?? "cli",
    id: "cli",
    note: argValue("--note"),
  });
}
