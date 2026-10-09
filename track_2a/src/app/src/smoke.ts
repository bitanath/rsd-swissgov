// TODO : - Headless self-check used by test/macos.sh and test/linux.sh.
import { generate, streamGenerate } from "./funcs";

const PROMPT = "Hello";

export function smokeRequested(): boolean {
  return process.argv.includes("--smoke") || Boolean(process.env.APERDIFF_SMOKE);
}

export async function runSmoke(): Promise<void> {
  console.log(`APERDIFF_SMOKE prompt: ${JSON.stringify(PROMPT)}`);

  const whole = generate({ prompt: PROMPT, temperature: 0.0, maxTokens: 128 });
  console.log(`  whole     tokens=${whole.tokens} chars=${whole.text.length}  ${whole.text.replace(/\s+/g, " ").trim().slice(0, 100)}`);

  let last = "";
  let streamed = "";
  const started = Date.now();
  const stream = await streamGenerate({
    prompt: PROMPT,
    temperature: 0.0,
    maxTokens: 128,
    onChunk: (text) => {
      process.stdout.write(text.slice(last.length));
      last = text;
      streamed = text;
    },
  });
  process.stdout.write("\n");

  const elapsed = Date.now() - started;
  const tps = elapsed > 0 ? stream.tokens / (elapsed / 1000) : 0;
  console.log(
    `  streaming tokens=${stream.tokens} chars=${streamed.length} truncated=${stream.truncated} ms=${elapsed} avg_tok_s=${tps.toFixed(2)}`,
  );

  const match = whole.text === streamed;
  console.log(`APERDIFF_SMOKE: nonstream==stream=${match}`);

  if (streamed.trim().length === 0 || !match) process.exitCode = 1;
}
