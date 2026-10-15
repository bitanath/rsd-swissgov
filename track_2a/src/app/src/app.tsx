import { createCliRenderer } from "@opentui/core";
import { createRoot } from "@opentui/react";

import { ReviewScreen } from "./interface/review-screen";
import { Toaster } from "./interface/toaster";
import type { HunkDiffFileInput } from "./libraries/hunk/opentui";
import { HunkThemeProvider } from "./libraries/theme";

export function render(opts: { initialFile?: HunkDiffFileInput } = {}): void {
  const start = async () => {
    const renderer = await createCliRenderer({ useMouse: true });
    createRoot(renderer).render(
      <HunkThemeProvider>
        <Toaster>
          <ReviewScreen initialFile={opts.initialFile} />
        </Toaster>
      </HunkThemeProvider>,
    );
  };
  void start();
}
