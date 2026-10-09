import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

import { ThemeContext } from "../theme/use-theme";
import type { Theme } from "../theme/types";
import { HUNK_DIFF_THEME_NAMES, type HunkDiffThemeName } from "./hunk/opentui/themes";
import { resolveTheme, type AppTheme } from "./hunk/ui/themes";

const CONFIG_DIR = join(homedir(), ".aperdiff");
const THEME_FILE = join(CONFIG_DIR, "hunk-theme.json");

export const DEFAULT_HUNK_THEME: HunkDiffThemeName = "github-dark-default";

export function isHunkThemeName(name: string | null | undefined): name is HunkDiffThemeName {
  return Boolean(name) && (HUNK_DIFF_THEME_NAMES as readonly string[]).includes(name as string);
}

export function loadHunkThemeName(): HunkDiffThemeName {
  const env = process.env.APERDIFF_HUNK_THEME;
  if (isHunkThemeName(env)) return env;
  try {
    const parsed = JSON.parse(readFileSync(THEME_FILE, "utf8")) as { theme?: string };
    if (isHunkThemeName(parsed.theme)) return parsed.theme;
  } catch {
    // XXX No config yet (or unreadable) — fall through to the default.
    console.error("No theme config yet")
  }
  return DEFAULT_HUNK_THEME;
}

export function saveHunkThemeName(name: HunkDiffThemeName): void {
  try {
    if (!existsSync(CONFIG_DIR)) mkdirSync(CONFIG_DIR, { recursive: true, mode: 0o700 });
    writeFileSync(THEME_FILE, `${JSON.stringify({ theme: name }, null, 2)}\n`, { mode: 0o600 });
  } catch {
    // XXX Best-effort: a read-only home should not break the app.
    console.error("No dir config yet")
  }
}

/** Project the Hunk `AppTheme` onto the app `Theme` shape queen's chrome reads. */
function toAppTheme(theme: AppTheme): Theme {
  return {
    name: theme.id,
    colors: {
      primary: theme.accent,
      primaryForeground: theme.background,
      secondary: theme.accentMuted,
      secondaryForeground: theme.text,
      accent: theme.accent,
      accentForeground: theme.background,
      success: theme.addedSignColor,
      successForeground: theme.background,
      warning: theme.fileModified,
      warningForeground: theme.background,
      error: theme.removedSignColor,
      errorForeground: theme.background,
      info: theme.accent,
      infoForeground: theme.background,
      background: theme.background,
      foreground: theme.text,
      muted: theme.panelAlt,
      mutedForeground: theme.muted,
      border: theme.border,
      focusRing: theme.accent,
      selection: theme.selectedHunk,
      selectionForeground: theme.text,
    },
    spacing: { 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 6, 8: 8 },
    typography: { bold: true, sm: "", base: "", lg: "", xl: "" },
    border: { style: "rounded", color: theme.border, focusColor: theme.accent },
  };
}

interface HunkThemeContextValue {
  name: HunkDiffThemeName;
  theme: AppTheme;
  setTheme: (name: HunkDiffThemeName) => void;
  cycleTheme: () => void;
}

const HunkThemeContext = createContext<HunkThemeContextValue | null>(null);

export function HunkThemeProvider({
  children,
  initial,
}: {
  children: ReactNode;
  initial?: HunkDiffThemeName;
}) {
  const [name, setName] = useState<HunkDiffThemeName>(() => initial ?? loadHunkThemeName());
  const theme = useMemo(() => resolveTheme(name, null), [name]);
  const value = useMemo<HunkThemeContextValue>(
    () => ({
      name,
      theme,
      setTheme: (next) => {
        setName(next);
        saveHunkThemeName(next);
      },
      cycleTheme: () =>
        setName((current) => {
          const index = HUNK_DIFF_THEME_NAMES.indexOf(current);
          const next =
            HUNK_DIFF_THEME_NAMES[(index + 1) % HUNK_DIFF_THEME_NAMES.length] ?? DEFAULT_HUNK_THEME;
          saveHunkThemeName(next);
          return next;
        }),
    }),
    [name, theme],
  );
  const appTheme = useMemo(() => toAppTheme(theme), [theme]);
  const appThemeValue = useMemo(() => ({ theme: appTheme, setTheme: () => {} }), [appTheme]);

  return (
    <ThemeContext.Provider value={appThemeValue}>
      <HunkThemeContext.Provider value={value}>{children}</HunkThemeContext.Provider>
    </ThemeContext.Provider>
  );
}

export function useHunkTheme(): HunkThemeContextValue {
  const value = useContext(HunkThemeContext);
  if (!value) throw new Error("useHunkTheme must be used within HunkThemeProvider");
  return value;
}
