import { useMemo, useState } from "react";

import { Modal } from "../components/modal";
import { HUNK_DIFF_THEME_NAMES } from "../libraries/hunk/opentui/themes";
import { useHunkTheme } from "../libraries/theme";

export function ThemeModal({ onClose }: { onClose: () => void }) {
  const { name, theme, setTheme } = useHunkTheme();
  // Capture the theme active on open, once: preview must not move the revert target.
  const [original] = useState(name);

  const options = useMemo(
    () => HUNK_DIFF_THEME_NAMES.map((id) => ({ name: id, description: "", value: id })),
    [],
  );
  const selectedIndex = Math.max(0, HUNK_DIFF_THEME_NAMES.indexOf(name));
  // Hunk bundles ~65 themes; window the list so it never overflows the panel.
  const listHeight = Math.min(HUNK_DIFF_THEME_NAMES.length, 16);

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
            const next = HUNK_DIFF_THEME_NAMES[index];
            if (next) {
              setTheme(next);
              onClose();
            }
          }}
          onChange={(index: number) => {
            const next = HUNK_DIFF_THEME_NAMES[index];
            if (next) setTheme(next);
          }}
        />
      </box>
    </Modal>
  );
}
