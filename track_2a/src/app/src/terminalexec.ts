//TODO: Headless CLI for the judges: `predict` and `results`. The TUI stays in app.tsx.
import { pairScores } from "./funcs";
import { T_DIFF, T_GRU } from "./libraries/clit/analyze";
import { words } from "./libraries/clit/dataset";
import { pairScoresBothProgress } from "./libraries/progress/ffi";
import { type Align, naturalWidths, padCell } from "./libraries/table";
import swissgovJsonlPath from "./data/swissgov.jsonl" with { type: "file" };
import predictionsJsonlPath from "./data/predictions.jsonl" with { type: "file" };

export type HeadArg = "supervised" | "unsupervised" | "combined";
export const DEFAULT_HEAD: HeadArg = "supervised";
export const HEAD_ARGS: HeadArg[] = ["supervised", "unsupervised", "combined"];

export type Lang = "de" | "fr" | "it";
export const ALL_LANGS: Lang[] = ["de", "fr", "it"];

export interface ExecRow {
  id: string;
  text_a: string;
  text_b: string;
  labels_a: number[];
  labels_b: number[];
}

interface RowMeta {
  id: string;
  lang: Lang;
  verdict: "same" | "different";
  flagged_a: number;
  flagged_b: number;
}

// --- args -------------------------------------------------------------------
function argValue(flag: string): string | undefined {
  const i = process.argv.indexOf(flag);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

function argPresent(flag: string): boolean {
  return process.argv.includes(flag);
}

export function terminalExecRequested(): boolean {
  return process.argv.includes("predict") || process.argv.includes("results");
}

export function parseHead(raw?: string): HeadArg {
  return HEAD_ARGS.includes(raw as HeadArg) ? (raw as HeadArg) : DEFAULT_HEAD;
}

export function parseLangs(raw?: string): Lang[] {
  if (!raw || raw === "all") return ALL_LANGS;
  const wanted = new Set(raw.split(",").map((s) => s.trim()));
  const picked = ALL_LANGS.filter((l) => wanted.has(l));
  return picked.length > 0 ? picked : ALL_LANGS;
}

// --- jsonl ------------------------------------------------------------------
function parseJsonl(text: string): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (line) out.push(JSON.parse(line) as Record<string, unknown>);
  }
  return out;
}

function toExecRow(row: Record<string, unknown>): ExecRow {
  return {
    id: String(row["id"] ?? ""),
    text_a: String(row["text_a"] ?? ""),
    text_b: String(row["text_b"] ?? ""),
    labels_a: Array.isArray(row["labels_a"]) ? (row["labels_a"] as number[]) : [],
    labels_b: Array.isArray(row["labels_b"]) ? (row["labels_b"] as number[]) : [],
  };
}

export function langOf(row: Record<string, unknown>): Lang {
  const l = row["lang"];
  if (l === "de" || l === "fr" || l === "it") return l;
  const m = /^admin_(de|fr|it)_/.exec(String(row["id"] ?? ""));
  return (m?.[1] as Lang) ?? "de";
}

export async function readBundledInput(): Promise<Record<string, unknown>[]> {
  return parseJsonl(await Bun.file(swissgovJsonlPath).text());
}

export async function readBundledResults(): Promise<ExecRow[]> {
  return parseJsonl(await Bun.file(predictionsJsonlPath).text()).map(toExecRow);
}

// --- scoring ----------------------------------------------------------------
export interface ComputedRow {
  labels_a: number[];
  labels_b: number[];
  meta: RowMeta;
}

function thresholds(lang: Lang): { tg: number; td: number } {
  return { tg: T_GRU[lang] ?? 0.1, td: T_DIFF[lang] ?? 0.22 };
}

export function computeRow(textA: string, textB: string, head: HeadArg, lang: Lang): ComputedRow {
  const wcA = words(textA).length;
  const wcB = words(textB).length;
  const { tg, td } = thresholds(lang);
  const a: number[] = [];
  const b: number[] = [];
  let flaggedA = 0;
  let flaggedB = 0;

  if (head === "combined") {
    const both = pairScoresBothProgress(textA, textB, wcA, wcB);
    if (!both) throw new Error("pair_scores_both_ex failed");
    for (let i = 0; i < wcA; i++) {
      const g = both.gru.a[i] ?? 0;
      const d = both.diff.a[i] ?? 0;
      a.push(Math.max(g, d));
      if (g > tg || d > td) flaggedA++;
    }
    for (let i = 0; i < wcB; i++) {
      const g = both.gru.b[i] ?? 0;
      const d = both.diff.b[i] ?? 0;
      b.push(Math.max(g, d));
      if (g > tg || d > td) flaggedB++;
    }
  } else {
    const s = pairScores(textA, textB, head === "supervised" ? "gru" : "diffalign", wcA, wcB);
    if (!s) throw new Error("pair_scores failed");
    const cut = head === "supervised" ? tg : td;
    for (let i = 0; i < wcA; i++) {
      const v = s.a[i] ?? 0;
      a.push(v);
      if (v > cut) flaggedA++;
    }
    for (let i = 0; i < wcB; i++) {
      const v = s.b[i] ?? 0;
      b.push(v);
      if (v > cut) flaggedB++;
    }
  }

  const different = flaggedA + flaggedB > 0;
  return { labels_a: a, labels_b: b, meta: { id: "", lang, verdict: different ? "different" : "same", flagged_a: flaggedA, flagged_b: flaggedB } };
}

export async function predictRows(
  rows: Record<string, unknown>[],
  head: HeadArg,
  langs: Lang[],
  onDoc?: (done: number, total: number) => void,
): Promise<{ predictions: ExecRow[]; metas: RowMeta[] }> {
  const selected = rows.filter((r) => langs.includes(langOf(r)));
  const predictions: ExecRow[] = [];
  const metas: RowMeta[] = [];
  selected.forEach((row, i) => {
    const id = String(row["id"] ?? "");
    const lang = langOf(row);
    const computed = computeRow(String(row["text_a"] ?? ""), String(row["text_b"] ?? ""), head, lang);
    predictions.push({ id, text_a: String(row["text_a"] ?? ""), text_b: String(row["text_b"] ?? ""), labels_a: computed.labels_a, labels_b: computed.labels_b });
    metas.push({ ...computed.meta, id });
    onDoc?.(i + 1, selected.length);
  });
  return { predictions, metas };
}

// --- rendering --------------------------------------------------------------
export function renderMarkdownTable(headers: string[], rows: string[][], aligns?: Align[]): string {
  const widths = naturalWidths(headers, rows);
  const align = aligns ?? headers.map(() => "left" as Align);
  const renderRow = (cells: string[]): string =>
    `| ${cells.map((c, i) => padCell(c, widths[i] ?? 0, align[i])).join(" | ")} |`;
  const rule = `| ${widths
    .map((w, i) => {
      const bar = "-".repeat(Math.max(3, align[i] === "center" ? w - 2 : w));
      if (align[i] === "right") return `${bar}:`;
      if (align[i] === "center") return `:${bar}:`;
      return bar;
    })
    .join(" | ")} |`;
  return [renderRow(headers), rule, ...rows.map(renderRow)].join("\n");
}

const TABLE_HEADERS = ["id", "lang", "verdict", "flagged_a", "flagged_b"];
const TABLE_ALIGNS: Align[] = ["left", "left", "left", "right", "right"];

export function renderMetaTable(metas: RowMeta[]): string {
  const rows = metas.map((m) => [m.id, m.lang, m.verdict, String(m.flagged_a), String(m.flagged_b)]);
  const totalFlaggedA = metas.reduce((n, m) => n + m.flagged_a, 0);
  const totalFlaggedB = metas.reduce((n, m) => n + m.flagged_b, 0);
  const different = metas.filter((m) => m.verdict === "different").length;
  const footer = `\n\n${metas.length} docs · ${different} different · ${totalFlaggedA} flagged (a) · ${totalFlaggedB} flagged (b)`;
  return renderMarkdownTable(TABLE_HEADERS, rows, TABLE_ALIGNS) + footer;
}

export function metaFromPredictions(rows: ExecRow[]): RowMeta[] {
  return rows.map((r) => {
    const lang = langOf({ id: r.id });
    const tg = T_GRU[lang] ?? 0.1;
    const flaggedA = r.labels_a.reduce((n, v) => n + (v > tg ? 1 : 0), 0);
    const flaggedB = r.labels_b.reduce((n, v) => n + (v > tg ? 1 : 0), 0);
    return { id: r.id, lang, verdict: flaggedA + flaggedB > 0 ? "different" : "same", flagged_a: flaggedA, flagged_b: flaggedB };
  });
}

function jsonlOf(rows: ExecRow[]): string {
  return rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
}

function outputPathOf(name: string): string {
  return `${name.replace(/\.[^./\\]*$/, "")}.jsonl`;
}

// --- entry point ------------------------------------------------------------
function log(message: string): void {
  process.stderr.write(`${message}\n`);
}

export async function runTerminalExec(): Promise<void> {
  const results = process.argv.includes("results");
  const langs = parseLangs(argValue("--lang"));
  const asJson = argPresent("--json");

  if (results) {
    const rows = (await readBundledResults()).filter((r) => langs.includes(langOf({ id: r.id })));
    const output = asJson ? jsonlOf(rows) : renderMetaTable(metaFromPredictions(rows));
    process.stdout.write(`${output}\n`);
    return;
  }

  const head = parseHead(argValue("--head"));
  const input = argValue("--input");
  const rows = input ? parseJsonl(await Bun.file(input).text()) : await readBundledInput();

  const startedAt = Date.now();
  log(`predict: head=${head} lang=${langs.join(",")} docs=${rows.filter((r) => langs.includes(langOf(r))).length} input=${input ?? "bundled:data/swissgov.jsonl"}`);
  const { predictions, metas } = await predictRows(rows, head, langs, (done, total) => {
    if (done % 10 === 0 || done === total) log(`  ${done}/${total} (${(Date.now() - startedAt) / 1000 | 0}s)`);
  });

  const outName = argValue("--output");
  if (outName) {
    const path = outputPathOf(outName);
    await Bun.write(path, jsonlOf(predictions));
    log(`wrote ${predictions.length} predictions -> ${path}`);
  }

  process.stdout.write(`${asJson ? jsonlOf(predictions) : renderMetaTable(metas)}\n`);
}
