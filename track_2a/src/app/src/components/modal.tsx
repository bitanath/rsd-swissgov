// Ported from termcn `opentui/modal`
import { useKeyboard, useTerminalDimensions } from "@opentui/react";
import type { ReactNode } from "react";

import { useTheme } from "../theme/use-theme";

export function Modal({
  title,
  width = 64,
  height = 18,
  onClose,
  children,
}: {
  title: string;
  width?: number;
  height?: number;
  onClose: () => void;
  children: ReactNode;
}) {
  const theme = useTheme();
  const { width: cols, height: rows } = useTerminalDimensions();
  const panelW = Math.min(width, Math.max(24, cols - 4));
  const panelH = Math.min(height, Math.max(6, rows - 2));

  useKeyboard((key) => {
    if (key.name === "escape") {
      key.preventDefault();
      onClose();
    }
  });

  return (
    <box
      position="absolute"
      top={0}
      left={0}
      width={cols}
      height={rows}
      zIndex={100}
      justifyContent="center"
      alignItems="center"
    >
      {/* Dim layer is a separate absolute sibling so opacity does not dim the panel. */}
      <box
        position="absolute"
        top={0}
        left={0}
        width={cols}
        height={rows}
        backgroundColor={theme.colors.background}
        opacity={0.85}
      />
      <box
        width={panelW}
        height={panelH}
        flexDirection="column"
        borderStyle={theme.border.style}
        borderColor={theme.colors.focusRing}
        backgroundColor={theme.colors.background}
        title={` ${title} (esc) `}
        titleColor={theme.colors.primary}
        titleAlignment="center"
        paddingLeft={2}
        paddingRight={2}
        paddingTop={1}
        paddingBottom={1}
        zIndex={101}
      >
        {children}
      </box>
    </box>
  );
}
