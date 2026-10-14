import { buildCliFile } from "./src/libraries/clit/cli";
import { runSmoke, smokeRequested } from "./src/smoke";
import { runSpeedtest, speedtestRequested } from "./src/speedtest";
import { runTerminalExec, terminalExecRequested } from "./src/terminalexec";

if (speedtestRequested()) {
  await runSpeedtest();
} else if (smokeRequested()) {
  await runSmoke();
} else if (terminalExecRequested()) {
  await runTerminalExec();
} else {
  const { render } = await import("./src/app");
  render({ initialFile: buildCliFile() });
}
