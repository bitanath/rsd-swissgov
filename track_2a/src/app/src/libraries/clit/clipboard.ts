import { createClipboard, createHostClipboard, createRendererClipboardAdapter, } from "@opentui/core";
import { useRenderer, useSelectionHandler } from "@opentui/react";
import { useEffect, useMemo } from "react";

export function useCopyOnSelect(onResult: (copied: boolean) => void): void {
  const renderer = useRenderer();
  const clipboard = useMemo(
    () =>
      createClipboard({
        host: createHostClipboard(),
        terminal: createRendererClipboardAdapter(renderer),
      }),
    [renderer],
  );

  useEffect(() => {
    return () => {
      void clipboard.dispose();
    };
  }, [clipboard]);

  useSelectionHandler((selection) => {
    const text = selection.getSelectedText();
    if (!text.trim()) {
      return;
    }
    void clipboard
      .writeText(text, { destination: "best-available" })
      .then((result) => {
        onResult(result.host.status === "written" || result.terminal.status === "attempted");
      })
      .catch(() => onResult(false));
  });
}
