import { buildCliFile } from "./src/libraries/clit/cli";
import { runSmoke, smokeRequested } from "./src/smoke";
import { runSpeedtest, speedtestRequested } from "./src/speedtest";

if (speedtestRequested()) {
  await runSpeedtest();
} else if (smokeRequested()) {
  await runSmoke();
} else {
  const { render } = await import("./src/app");
  render({ initialFile: buildCliFile() });
}
