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

function log(message: string): void {
  process.stderr.write(`${message}\n`);
}

const TICK_REQUEST_NS = 250_000_000; // ask for a tick at most every 250ms

function run(prompt: string, maxTokens: number, onTick?: (generated: number, maxTokens: number) => void): Run {
  const t0 = Bun.nanoseconds();
  const handle = startStream({ prompt, maxTokens, temperature: 0 });
  const out = new Uint8Array(65536);
  let lastTick = 0;
  try {
    for (;;) {
      const n = pollStream(handle, out);
      if (n === -2) throw new EngineFailed();
      if (n === -1) break;
      if (n > 0 && onTick) {
        const now = Bun.nanoseconds();
        if (now - lastTick >= TICK_REQUEST_NS) {
          lastTick = now;
          onTick(streamTimings(handle).generated, maxTokens);
        }
      }
      if (n === 0) Bun.sleepSync(1); // idle: tiny sleep, well under one token interval
    }
    const totalMs = (Bun.nanoseconds() - t0) / 1e6;
    return { ...streamTimings(handle), totalMs };
  } finally {
    freeStream(handle);
  }
}

//FIXME: Live decode counter: rewritten in place on a TTY, throttled to periodic lines when stderr is captured (docker logs, pipes, CI).
function makeCounter(): { update: (generated: number, maxTokens: number) => void; finish: () => void } {
  const tty = Boolean(process.stderr.isTTY);
  const t0 = Date.now();
  let lastLogged = 0;
  const update = (generated: number, maxTokens: number): void => {
    const now = Date.now();
    if (!tty && now - lastLogged < 2000) return;
    lastLogged = now;
    const secs = Math.max((now - t0) / 1000, 1e-3);
    const rate = generated / secs;
    const text = `  decode ${generated}/${maxTokens} tok · ${rate.toFixed(1)} tok/s`;
    if (tty) process.stderr.write(`\r${text}`);
    else process.stderr.write(`${text}\n`);
  };
  const finish = (): void => {
    if (tty) process.stderr.write("\n");
  };
  return { update, finish };
}

const rssMb = () => process.memoryUsage().rss / (1024 * 1024);

export async function runSpeedtest(): Promise<void> {
  log("speedtest: loading model + warmup (this is the long part)");
  const warmStart = Bun.nanoseconds();
  run("Hello", WARMUP_TOKENS); // warmup: model load (mmap) + page-in + allocations
  const warmupMs = (Bun.nanoseconds() - warmStart) / 1e6;
  log(`speedtest: warmup done in ${warmupMs.toFixed(0)}ms (rss ${rssMb().toFixed(0)}MB)`);

  log(`speedtest: prefill (prompt ${PREFILL_PROMPT.length} chars)…`);
  const pre = run(PREFILL_PROMPT, PREFILL_MAX_TOKENS);
  const prefill = pre.ttftMs > 0 ? pre.promptTokens / (pre.ttftMs / 1000) : 0;
  log(`speedtest: prefill ${prefill.toFixed(1)} tok/s (prompt ${pre.promptTokens} tok, ttft ${pre.ttftMs.toFixed(1)}ms)`);

  log(`speedtest: decode up to ${DECODE_MAX_TOKENS} tokens…`);
  const counter = makeCounter();
  const dec = run(DECODE_PROMPT, DECODE_MAX_TOKENS, (generated, max) => counter.update(generated, max));
  counter.finish();
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
