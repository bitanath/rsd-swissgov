// TODO : - Headless self-check used by test/macos.sh and test/linux.sh.
// Full: LLM generate/stream + `results` + a tiny `predict` capture.
// Fast (APERDIFF_SMOKE_FAST=1 or --smoke-fast): skips the model, checks the CLI data paths only.
import { appendFileSync } from "node:fs";
import { tmpdir } from "node:os";

import { generate, streamGenerate } from "./funcs";
import { words } from "./libraries/clit/dataset";
import {
  bundledGoldById,
  metaFromPredictions,
  predictRows,
  readBundledResults,
  renderMetaTable,
} from "./terminalexec";

const PROMPT = "Hello";
const LOG_PATH = process.env.APERDIFF_SMOKE_LOG ?? `${tmpdir()}/aperdiff-smoke.log`;

export function smokeRequested(): boolean {
  return process.argv.includes("--smoke") || Boolean(process.env.APERDIFF_SMOKE);
}

function fastMode(): boolean {
  return process.argv.includes("--smoke-fast") || Boolean(process.env.APERDIFF_SMOKE_FAST);
}

let failures = 0;

function say(line = ""): void {
  appendFileSync(LOG_PATH, `${line}\n`);
  console.log(line);
}

function check(label: string, ok: boolean, detail = ""): void {
  if (!ok) failures += 1;
  say(`  [${ok ? "PASS" : "FAIL"}] ${label}${detail ? ` — ${detail}` : ""}`);
}

// Small, self-contained sample so no external data is needed.
const SAMPLE = [
  { id: "admin_de_smoke_0", text_a: "The cat sat on the mat .", text_b: "Die Katze sass auf der Matte ." },
  { id: "admin_fr_smoke_0", text_a: "The quick brown fox .", text_b: "Le renard brun rapide ." },
];

async function checkBundledResults(): Promise<void> {
  say("== results (bundled predictions.jsonl) ==");
  const all = await readBundledResults();
  check("bundled predictions parsed", all.length === 672, `${all.length} rows`);

  const de = all.filter((r) => r.id.startsWith("admin_de_"));
  check("lang filter (de)", de.length === 224, `${de.length} rows`);

  const metas = metaFromPredictions(de, await bundledGoldById());
  const table = renderMetaTable(metas);
  check("markdown table has header + rows", table.split("\n").length >= 3, `${table.split("\n").length} lines`);
  const header = table.split("\n")[0] ?? "";
  check("results table exposes spearman + kendall", header.includes("spearman") && header.includes("kendall"));
  const first = metas[0];
  check(
    "first de doc scored (rho, tau-b finite)",
    first !== undefined && first.spearman !== null && first.kendall !== null && Number.isFinite(first.spearman) && Number.isFinite(first.kendall),
    first ? `ρ=${first.spearman?.toFixed(3)} τ-b=${first.kendall?.toFixed(3)}` : "",
  );
  say(table.split("\n").slice(0, 4).join("\n"));
  say(`  … (${de.length} de rows rendered)`);

  const json = all.map((r) => JSON.stringify(r)).join("\n");
  let parsed = true;
  try {
    for (const line of json.split("\n")) JSON.parse(line);
  } catch {
    parsed = false;
  }
  check("--json rows are valid jsonl", parsed);
}

async function checkLlm(): Promise<void> {
  say("== LLM generate/stream ==");
  const whole = generate({ prompt: PROMPT, temperature: 0.0, maxTokens: 128 });
  check("generate returns text", whole.text.trim().length > 0, `tokens=${whole.tokens}`);
  let last = "";
  let streamed = "";
  const stream = await streamGenerate({
    prompt: PROMPT,
    temperature: 0.0,
    maxTokens: 128,
    onChunk: (text) => {
      last = text;
      streamed = text;
    },
  });
  void last;
  check("stream nonempty", streamed.trim().length > 0, `tokens=${stream.tokens}`);
  check("nonstream == stream", whole.text === streamed);
}

async function checkPredict(): Promise<void> {
  say("== predict (tiny sample, head=combined) ==");
  const { predictions } = await predictRows(SAMPLE, "combined", ["de", "fr"]);
  check("predict returned rows", predictions.length === 2, `${predictions.length} rows`);
  for (const row of predictions) {
    const na = words(row.text_a).length;
    const nb = words(row.text_b).length;
    check(`${row.id}: labels_a length == tokens`, row.labels_a.length === na, `${row.labels_a.length}/${na}`);
    check(`${row.id}: labels_b length == tokens`, row.labels_b.length === nb, `${row.labels_b.length}/${nb}`);
    check(`${row.id}: labels finite`, row.labels_a.every((v) => Number.isFinite(v)) && row.labels_b.every((v) => Number.isFinite(v)));
  }
}

export async function runSmoke(): Promise<void> {
  const fast = fastMode();
  say(`APERDIFF_SMOKE mode=${fast ? "fast" : "full"} platform=${process.platform}/${process.arch}`);
  say(`SMOKE LOG: ${LOG_PATH}`);
  say();

  try {
    await checkBundledResults();
    say();
    if (!fast) {
      await checkLlm();
      say();
      await checkPredict();
      say();
    }
  } catch (err) {
    failures += 1;
    say(`  [FAIL] uncaught: ${(err as Error).message}`);
  }

  say(`APERDIFF_SMOKE: ${failures === 0 ? "OK" : `${failures} FAILURE(S)`}`);
  if (failures > 0) process.exitCode = 1;
}
