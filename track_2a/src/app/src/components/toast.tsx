// Ported from termcn `opentui/toast`: an auto-dismissing notification with icon
import { useEffect, useState } from "react";

import { useTheme } from "../theme/use-theme";

export type ToastVariant = "success" | "error" | "warning" | "info";

const ICONS: Record<ToastVariant, string> = {
  error: "✗",
  info: "ℹ",
  success: "✓",
  warning: "⚠",
};

export interface ToastProps {
  message: string;
  variant?: ToastVariant;
  duration?: number;
  onDismiss?: () => void;
  icon?: string;
}

const BAR_WIDTH = 20;
const TICK_MS = 100;

export const Toast = ({
  message,
  variant = "info",
  duration = 3000,
  onDismiss,
  icon,
}: ToastProps) => {
  const theme = useTheme();
  const [elapsed, setElapsed] = useState(0);
  const [dismissed, setDismissed] = useState(false);

  const variantColor = (() => {
    switch (variant) {
      case "success":
        return theme.colors.success;
      case "error":
        return theme.colors.error;
      case "warning":
        return theme.colors.warning;
      default:
        return theme.colors.info;
    }
  })();

  useEffect(() => {
    const id = setTimeout(() => {
      setDismissed(true);
      onDismiss?.();
    }, duration);
    return () => clearTimeout(id);
  }, [duration, onDismiss]);

  useEffect(() => {
    if (dismissed) {
      return;
    }
    const id = setInterval(() => {
      setElapsed((e) => Math.min(e + TICK_MS, duration));
    }, TICK_MS);
    return () => clearInterval(id);
  }, [dismissed, duration]);

  if (dismissed) {
    return null;
  }

  const remaining = Math.max(0, duration - elapsed);
  const remainingSeconds = (remaining / 1000).toFixed(1);
  const progress = remaining / duration;
  const filledChars = Math.round(progress * BAR_WIDTH);
  const emptyChars = BAR_WIDTH - filledChars;
  const bar = "█".repeat(filledChars) + "░".repeat(emptyChars);

  const resolvedIcon = icon ?? ICONS[variant];

  return (
    <box
      borderStyle="rounded"
      borderColor={variantColor}
      backgroundColor={theme.colors.background}
      paddingLeft={1}
      paddingRight={1}
      paddingTop={0}
      paddingBottom={0}
      flexDirection="column"
    >
      <box flexDirection="row" gap={1}>
        <text fg={variantColor} selectable={false}>
          <b>{resolvedIcon}</b>
        </text>
        <text selectable={false}>{message}</text>
      </box>
      <box flexDirection="row" gap={1}>
        <text fg={variantColor} selectable={false}>{bar}</text>
        <text fg={theme.colors.muted} selectable={false}>{`${remainingSeconds}s`}</text>
      </box>
    </box>
  );
};
