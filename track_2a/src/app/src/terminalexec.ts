//TODO: Headless CLI for the judges: `predict` and `results`. The TUI stays in app.tsx.
import { ensureModel } from "./funcs";
import { T_DIFF, T_GRU } from "./libraries/clit/analyze";
import { words } from "./libraries/clit/dataset";
import { docMetric } from "./libraries/clit/metrics";
import { pairScoresBothProgress, pairScoresProgress, type HeadProgress } from "./libraries/progress/ffi";
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
  spearman?: number | null;
  kendall?: number | null;
}

interface RowMeta {
  id: string;
  lang: Lang;
  verdict: "same" | "different";
  flagged_a: number;
  flagged_b: number;
  spearman: number | null;
  kendall: number | null;
}

export interface GoldRow {
  labels_a: number[];
  labels_b: number[];
}

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

export function splitOf(row: Record<string, unknown>): string | undefined {
  const s = row["split"];
  return typeof s === "string" ? s : undefined;
}

let bundledTestIdsCache: Set<string> | undefined;

// The scored split is the benchmark test split. Rows carry `split` when they come
// from the bundled data; `results` reads predictions.jsonl (no `split`), so fall
// back to membership in the bundled test-id set.
export async function bundledTestIds(): Promise<Set<string>> {
  if (!bundledTestIdsCache) {
    const rows = await readBundledInput();
    bundledTestIdsCache = new Set(
      rows.filter((r) => r["split"] === "test").map((r) => String(r["id"] ?? "")),
    );
  }
  return bundledTestIdsCache;
}

export function filterTestRows(
  rows: Record<string, unknown>[],
  testIds: Set<string>,
): Record<string, unknown>[] {
  return rows.filter((r) => {
    const s = splitOf(r);
    return s !== undefined ? s === "test" : testIds.has(String(r["id"] ?? ""));
  });
}

let bundledGoldCache: Map<string, GoldRow> | undefined;

/** Bundled gold labels (from data/swissgov.jsonl) keyed by document id. */
export async function bundledGoldById(): Promise<Map<string, GoldRow>> {
  if (!bundledGoldCache) {
    const rows = await readBundledInput();
    bundledGoldCache = new Map();
    for (const r of rows) {
      bundledGoldCache.set(String(r["id"] ?? ""), {
        labels_a: Array.isArray(r["labels_a"]) ? (r["labels_a"] as number[]).map(Number) : [],
        labels_b: Array.isArray(r["labels_b"]) ? (r["labels_b"] as number[]).map(Number) : [],
      });
    }
  }
  return bundledGoldCache;
}

function goldOfRow(row: Record<string, unknown>): GoldRow | null {
  const a = Array.isArray(row["labels_a"]) ? (row["labels_a"] as number[]).map(Number) : [];
  const b = Array.isArray(row["labels_b"]) ? (row["labels_b"] as number[]).map(Number) : [];
  if (a.length === 0 && b.length === 0) return null;
  return { labels_a: a, labels_b: b };
}

function metricColumns(predA: number[], predB: number[], gold: GoldRow | null): { spearman: number | null; kendall: number | null } {
  if (!gold) return { spearman: null, kendall: null };
  const m = docMetric(predA, predB, gold.labels_a, gold.labels_b);
  return { spearman: Number.isNaN(m.spearman) ? null : m.spearman, kendall: Number.isNaN(m.kendall) ? null : m.kendall };
}

export interface ComputedRow {
  labels_a: number[];
  labels_b: number[];
  meta: RowMeta;
}

function thresholds(lang: Lang): { tg: number; td: number } {
  return { tg: T_GRU[lang] ?? 0.1, td: T_DIFF[lang] ?? 0.22 };
}

export function computeRow(textA: string, textB: string, head: HeadArg, lang: Lang, onSub?: HeadProgress): ComputedRow {
  const wcA = words(textA).length;
  const wcB = words(textB).length;
  const { tg, td } = thresholds(lang);
  const a: number[] = [];
  const b: number[] = [];
  let flaggedA = 0;
  let flaggedB = 0;

  if (head === "combined") {
    const both = pairScoresBothProgress(textA, textB, wcA, wcB, onSub);
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
    const s = pairScoresProgress(textA, textB, head === "supervised" ? "gru" : "diffalign", wcA, wcB, onSub);
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
  return { labels_a: a, labels_b: b, meta: { id: "", lang, verdict: different ? "different" : "same", flagged_a: flaggedA, flagged_b: flaggedB, spearman: null, kendall: null } };
}

export async function predictRows(
  rows: Record<string, unknown>[],
  head: HeadArg,
  langs: Lang[],
  onDoc?: (done: number, total: number) => void,
  onSub?: HeadProgress,
): Promise<{ predictions: ExecRow[]; metas: RowMeta[] }> {
  const selected = rows.filter((r) => langs.includes(langOf(r)));
  const predictions: ExecRow[] = [];
  const metas: RowMeta[] = [];
  selected.forEach((row, i) => {
    const id = String(row["id"] ?? "");
    const lang = langOf(row);
    const computed = computeRow(String(row["text_a"] ?? ""), String(row["text_b"] ?? ""), head, lang, onSub);
    const metrics = metricColumns(computed.labels_a, computed.labels_b, goldOfRow(row));
    predictions.push({ id, text_a: String(row["text_a"] ?? ""), text_b: String(row["text_b"] ?? ""), labels_a: computed.labels_a, labels_b: computed.labels_b, ...metrics });
    metas.push({ ...computed.meta, id, ...metrics });
    onDoc?.(i + 1, selected.length);
  });
  return { predictions, metas };
}

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

const TABLE_HEADERS = ["id", "lang", "verdict", "flagged_a", "flagged_b", "spearman", "kendall"];
const TABLE_ALIGNS: Align[] = ["left", "left", "left", "right", "right", "right", "right"];

function fmtMetric(v: number | null): string {
  return v === null || Number.isNaN(v) ? "—" : v.toFixed(3);
}

export function renderMetaTable(metas: RowMeta[]): string {
  const rows = metas.map((m) => [m.id, m.lang, m.verdict, String(m.flagged_a), String(m.flagged_b), fmtMetric(m.spearman), fmtMetric(m.kendall)]);
  const totalFlaggedA = metas.reduce((n, m) => n + m.flagged_a, 0);
  const totalFlaggedB = metas.reduce((n, m) => n + m.flagged_b, 0);
  const different = metas.filter((m) => m.verdict === "different").length;
  const footer = `\n\n${metas.length} docs · ${different} different · ${totalFlaggedA} flagged (a) · ${totalFlaggedB} flagged (b)`;
  return renderMarkdownTable(TABLE_HEADERS, rows, TABLE_ALIGNS) + footer;
}

export function metaFromPredictions(rows: ExecRow[], goldById?: Map<string, GoldRow>): RowMeta[] {
  return rows.map((r) => {
    const lang = langOf({ id: r.id });
    const tg = T_GRU[lang] ?? 0.1;
    const flaggedA = r.labels_a.reduce((n, v) => n + (v > tg ? 1 : 0), 0);
    const flaggedB = r.labels_b.reduce((n, v) => n + (v > tg ? 1 : 0), 0);
    const metrics = metricColumns(r.labels_a, r.labels_b, goldById?.get(r.id) ?? null);
    return { id: r.id, lang, verdict: flaggedA + flaggedB > 0 ? "different" : "same", flagged_a: flaggedA, flagged_b: flaggedB, ...metrics };
  });
}

function jsonlOf(rows: ExecRow[]): string {
  return rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
}

function outputPathOf(name: string): string {
  return `${name.replace(/\.[^./\\]*$/, "")}.jsonl`;
}

interface Progress {
  update: (done: number) => void;
  sub: (phase: number, done: number, total: number) => void;
  finish: () => void;
}

function makeProgress(label: string, total: number): Progress {
  const tty = Boolean(process.stderr.isTTY);
  const width = 24;
  const t0 = Date.now();
  let lastLogged = 0;
  let done = 0;
  let subFrac = 0;
  let subPhase = 0;

  const line = (): string => {
    const now = Date.now();
    const elapsed = (now - t0) / 1000;
    const frac = total > 0 ? done / total : 1;
    const pct = Math.round(frac * 100);
    const eta = done > 0 ? (elapsed * (total - done)) / done : 0;
    const filled = Math.round(frac * width);
    const bar = tty ? ` [${"#".repeat(filled)}${"-".repeat(Math.max(0, width - filled))}]` : "";
    const sub = subFrac > 0 && done < total ? ` · doc ${Math.round(subFrac * 100)}%${subPhase === 2 ? " (classifying)" : ""}` : "";
    return `${label}${bar} ${done}/${total} ${pct}%${sub} · ${elapsed | 0}s eta ${eta | 0}s`;
  };

  const draw = (force: boolean): void => {
    const now = Date.now();
    if (tty) {
      process.stderr.write(`\r${line()}`);
    } else if (force || now - lastLogged >= 2000) {
      lastLogged = now;
      process.stderr.write(`${line()}\n`);
    }
  };

  const update = (d: number): void => {
    done = d;
    subFrac = 0;
    subPhase = 0;
    draw(d === total);
  };

  const sub = (phase: number, d: number, t: number): void => {
    subPhase = phase;
    subFrac = t > 0 ? d / t : 0;
    draw(false);
  };

  const finish = (): void => {
    if (tty) process.stderr.write("\n");
  };

  return { update, sub, finish };
}

function log(message: string): void {
  process.stderr.write(`${message}\n`);
}

export async function runTerminalExec(): Promise<void> {
  const results = process.argv.includes("results");
  const langs = parseLangs(argValue("--lang"));
  const asJson = argPresent("--json");
  const testOnly = argPresent("--test");

  if (results) {
    let rows = (await readBundledResults()).filter((r) => langs.includes(langOf({ id: r.id })));
    if (testOnly) {
      const testIds = await bundledTestIds();
      rows = rows.filter((r) => testIds.has(r.id));
    }
    const metas = metaFromPredictions(rows, await bundledGoldById());
    const output = asJson
      ? jsonlOf(rows.map((r, i) => ({ ...r, spearman: metas[i]!.spearman, kendall: metas[i]!.kendall })))
      : renderMetaTable(metas);
    process.stdout.write(`${output}\n`);
    return;
  }

  const head = parseHead(argValue("--head"));
  const input = argValue("--input");
  const rowsAll = input ? parseJsonl(await Bun.file(input).text()) : await readBundledInput();
  const rows = testOnly ? filterTestRows(rowsAll, await bundledTestIds()) : rowsAll;

  const selected = rows.filter((r) => langs.includes(langOf(r))).length;
  log(`predict: head=${head} lang=${langs.join(",")}${testOnly ? " test" : ""} docs=${selected} input=${input ?? "bundled:data/swissgov.jsonl"}`);

  log("  loading engine (first run may be a little slow)");
  ensureModel();

  const bar = makeProgress("predict", selected);
  bar.update(0);
  const { predictions, metas } = await predictRows(
    rows,
    head,
    langs,
    (done) => bar.update(done),
    (phase, done, total) => bar.sub(phase, done, total),
  );
  bar.finish();

  const outName = argValue("--output");
  if (outName) {
    const path = outputPathOf(outName);
    await Bun.write(path, jsonlOf(predictions));
    log(`wrote ${predictions.length} predictions -> ${path}`);
  }

  process.stdout.write(`${asJson ? jsonlOf(predictions) : renderMetaTable(metas)}\n`);
}
