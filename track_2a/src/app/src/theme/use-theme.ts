import { createContext, useContext } from "react";
import type { Theme, ThemeContextValue } from "./types";

const FALLBACK_THEME: Theme = {
  name: "fallback",
  colors: {
    primary: "#58a6ff",
    primaryForeground: "#0d1117",
    secondary: "#1f6feb",
    secondaryForeground: "#e6edf3",
    accent: "#58a6ff",
    accentForeground: "#0d1117",
    success: "#3fb950",
    successForeground: "#0d1117",
    warning: "#d29922",
    warningForeground: "#0d1117",
    error: "#f85149",
    errorForeground: "#0d1117",
    info: "#58a6ff",
    infoForeground: "#0d1117",
    background: "#0d1117",
    foreground: "#e6edf3",
    muted: "#161b22",
    mutedForeground: "#8b949e",
    border: "#30363d",
    focusRing: "#58a6ff",
    selection: "#264f78",
    selectionForeground: "#e6edf3",
  },
  spacing: { 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 6, 8: 8 },
  typography: { bold: true, sm: "", base: "", lg: "", xl: "" },
  border: { style: "rounded", color: "#30363d", focusColor: "#58a6ff" },
};

export const ThemeContext = createContext<ThemeContextValue>({
  setTheme: () => {},
  theme: FALLBACK_THEME,
});

export const useTheme = (): Theme => useContext(ThemeContext).theme;

export const useThemeUpdater = (): ((theme: Theme) => void) =>
  useContext(ThemeContext).setTheme;
