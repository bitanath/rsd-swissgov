import { useMemo, useState } from "react";

import { Modal } from "../components/modal";
import { SELECTABLE_THEMES, useHunkTheme } from "../libraries/theme";

export function ThemeModal({ onClose }: { onClose: () => void }) {
  const { name, theme, setTheme } = useHunkTheme();
  //FIXME: Capture the theme active on open, once: preview must not move the revert target.
  const [original] = useState(name);

  const options = useMemo(
    () => SELECTABLE_THEMES.map((id) => ({ name: id, description: "", value: id })),
    [],
  );
  const selectedIndex = Math.max(0, SELECTABLE_THEMES.indexOf(name));
  // FIXME: Curated list (github / catppuccin / vitesse); window it so it never overflows.
  const listHeight = Math.min(SELECTABLE_THEMES.length, 16);

  const revertAndClose = () => {
    setTheme(original);
    onClose();
  };

  return (
    <Modal
      title="theme"
      width={56}
      height={Math.min(listHeight + 6, 22)}
      onClose={revertAndClose}
    >
      <text fg={theme.muted}>Arrows preview · enter saves · esc cancels</text>
      <box flexGrow={1} minHeight={0} marginTop={1}>
        <select
          focused
          flexGrow={1}
          options={options}
          height={listHeight}
          selectedIndex={selectedIndex}
          showSelectionIndicator
          onSelect={(index: number) => {
            const next = SELECTABLE_THEMES[index];
            if (next) {
              setTheme(next);
              onClose();
            }
          }}
          onChange={(index: number) => {
            const next = SELECTABLE_THEMES[index];
            if (next) setTheme(next);
          }}
        />
      </box>
    </Modal>
  );
}
