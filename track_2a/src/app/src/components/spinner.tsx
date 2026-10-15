//Ported from termcn `opentui/spinner`
import { TextAttributes } from "@opentui/core";
import { useEffect, useState } from "react";
import { useTheme } from "../theme/use-theme";

const SPINNERS = {
  dots: { interval: 80, frames: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] },
  earth: { interval: 180, frames: ["🌍 ", "🌎 ", "🌏 "] },
} as const;

export type SpinnerType = keyof typeof SPINNERS;

export function Spinner({
  type = "dots",
  label,
  color,
}: {
  type?: SpinnerType;
  label?: string;
  color?: string;
}) {
  const theme = useTheme();
  const { interval, frames } = SPINNERS[type];
  const [i, setI] = useState(0);
  useEffect(() => {
    setI(0);
    const t = setInterval(() => setI((v) => (v + 1) % frames.length), interval);
    return () => clearInterval(t);
  }, [frames, interval]);
  const glyph = frames[i] ?? frames[0] ?? "";
  const resolvedColor = color ?? theme.colors.mutedForeground;
  return (
    <text fg={resolvedColor} attributes={color ? undefined : TextAttributes.DIM}>
      {label ? `${glyph} ${label}` : glyph}
    </text>
  );
}
