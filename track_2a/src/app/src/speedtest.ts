import { EngineFailed, freeStream, pollStream, startStream, streamTimings } from "./funcs"; 

const MIN_DECODE = 15; //NOTE Standard is set at decode >= 15 tok/s, prefill >= 60 tok/s on M3 Pro 6P/6E cores
const MIN_PREFILL = 60;

const PREFILL_UNIT =
  "Apertus is an open, transparent large language model developed by the Swiss AI Initiative, designed for multilingual understanding and reliable, verifiable work across many domains and languages. ";
const PREFILL_PROMPT = PREFILL_UNIT.repeat(14);
const DECODE_PROMPT = "Write a detailed paragraph about the importance of open-source software in scientific research.";

const WARMUP_TOKENS = 4;
const PREFILL_MAX_TOKENS = 1;
const DECODE_MAX_TOKENS = 128;

export function speedtestRequested(): boolean {
  return (
    process.argv.includes("--speedtest") ||
    process.argv.includes("--bench") ||
    Boolean(process.env.APERDIFF_SPEEDTEST)
  );
}

interface Run {
  generated: number;
  promptTokens: number;
  ttftMs: number;
  totalMs: number;
}

function run(prompt: string, maxTokens: number): Run {
  const t0 = Bun.nanoseconds();
  const handle = startStream({ prompt, maxTokens, temperature: 0 });
  const out = new Uint8Array(65536);
  try {
    for (;;) {
      const n = pollStream(handle, out);
      if (n === -2) throw new EngineFailed();
      if (n === -1) break;
      if (n === 0) Bun.sleepSync(1); // idle: tiny sleep, well under one token interval
    }
    const totalMs = (Bun.nanoseconds() - t0) / 1e6;
    return { ...streamTimings(handle), totalMs };
  } finally {
    freeStream(handle);
  }
}

const rssMb = () => process.memoryUsage().rss / (1024 * 1024);

export async function runSpeedtest(): Promise<void> {
  const warmStart = Bun.nanoseconds();
  run("Hello", WARMUP_TOKENS); // warmup: model load (mmap) + page-in + allocations
  const warmupMs = (Bun.nanoseconds() - warmStart) / 1e6;

  const pre = run(PREFILL_PROMPT, PREFILL_MAX_TOKENS);
  const prefill = pre.ttftMs > 0 ? pre.promptTokens / (pre.ttftMs / 1000) : 0;

  const dec = run(DECODE_PROMPT, DECODE_MAX_TOKENS);
  const decodeWindowMs = Math.max(dec.totalMs - dec.ttftMs, 1);
  const decode = dec.generated / (decodeWindowMs / 1000);

  const pass = decode >= MIN_DECODE && prefill >= MIN_PREFILL;
  console.log(
    `SPEEDTEST apertus prefill=${prefill.toFixed(1)} decode=${decode.toFixed(1)} ` +
      `ttft_ms=${pre.ttftMs.toFixed(1)} prompt=${pre.promptTokens} generated=${dec.generated} ` +
      `warmup_ms=${warmupMs.toFixed(1)} rss_mb=${rssMb().toFixed(0)} ` +
      `${pass ? "PASS" : "FAIL"} (need decode>=${MIN_DECODE} prefill>=${MIN_PREFILL})`,
  );

  if (process.platform === "darwin" && !pass) process.exitCode = 1;
}
