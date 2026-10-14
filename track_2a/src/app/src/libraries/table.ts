export type Align = "left" | "right" | "center";

export interface ParsedTable {
  headers: string[];
  aligns: Align[];
  rows: string[][];
}

//MARK:- Table formatting logic ported from nvuillam/markdown-table-formatter
const DELIM_CELL = /^:?-{1,}:?$/;

const splitRow = (line: string): string[] => {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  const cells: string[] = [];
  let cur = "";
  for (let i = 0; i < s.length; i += 1) {
    const ch = s[i];
    if (ch === "\\" && s[i + 1] === "|") {
      cur += "|";
      i += 1;
    } else if (ch === "|") {
      cells.push(cur.trim());
      cur = "";
    } else {
      cur += ch;
    }
  }
  cells.push(cur.trim());
  return cells;
};

const alignOf = (cell: string): Align => {
  const left = cell.startsWith(":");
  const right = cell.endsWith(":");
  if (left && right) return "center";
  if (right) return "right";
  return "left";
};

export const parseMarkdownTable = (md: string): ParsedTable | null => {
  const lines = md
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length < 2) return null;

  const headers = splitRow(lines[0] ?? "");
  const delim = splitRow(lines[1] ?? "");
  if (delim.length === 0 || !delim.every((c) => DELIM_CELL.test(c))) return null;

  const aligns = delim.map(alignOf);
  const rows = lines.slice(2).map(splitRow);
  return { headers, aligns, rows };
};

export const naturalWidths = (headers: string[], rows: string[][]): number[] =>
  headers.map((header, i) => {
    let width = header.length;
    for (const row of rows) {
      width = Math.max(width, (row[i] ?? "").length);
    }
    return width;
  });

export const fitWidths = (
  natural: number[],
  total: number,
  min = 3,
): number[] => {
  const n = natural.length;
  if (n === 0) return [];

  const widths = natural.map((w) => Math.max(min, Math.floor(w)));
  const space = Math.max(n * min, Math.floor(total) - 3 * n - 1);
  let sum = widths.reduce((a, b) => a + b, 0);

  if (sum < space) {
    let extra = space - sum;
    let i = 0;
    while (extra > 0) {
      const idx = i % n;
      widths[idx] = (widths[idx] ?? min) + 1;
      extra -= 1;
      i += 1;
    }
  } else if (sum > space) {
    while (sum > space) {
      let widest = -1;
      for (let i = 0; i < n; i += 1) {
        const w = widths[i] ?? 0;
        if (w > min && (widest === -1 || w > (widths[widest] ?? 0))) widest = i;
      }
      if (widest === -1) break;
      widths[widest] = (widths[widest] ?? min) - 1;
      sum -= 1;
    }
  }

  return widths;
};

export const padCell = (
  value: string,
  width: number,
  align: Align = "left",
): string => {
  const s = String(value);
  if (s.length >= width) return s.slice(0, width);
  const diff = width - s.length;
  if (align === "right") return " ".repeat(diff) + s;
  if (align === "center") {
    const left = Math.floor(diff / 2);
    return " ".repeat(left) + s + " ".repeat(diff - left);
  }
  return s + " ".repeat(diff);
};

// MARK: - display-width aware helpers (CJK/emoji count as 2 columns)

const CONTROL_CHARS =
  /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]/g;


export const sanitize = (value: string): string => String(value).replace(CONTROL_CHARS, "");

export const displayWidth = (value: string): number => {
  const s = sanitize(value);
  const sw = (globalThis as { Bun?: { stringWidth?: (s: string) => number } }).Bun
    ?.stringWidth;
  return typeof sw === "function" ? sw(s) : s.length;
};

export const truncateDisplay = (value: string, width: number): string => {
  if (width <= 0) return "";
  const s = sanitize(value);
  if (displayWidth(s) <= width) return s;
  let out = "";
  let w = 0;
  for (const ch of s) {
    const cw = displayWidth(ch);
    if (w + cw > width) break;
    out += ch;
    w += cw;
  }
  return out;
};

export const ellipsizeDisplay = (value: string, width: number): string => {
  if (width <= 1) return "…";
  if (displayWidth(value) <= width) return value;
  return `${truncateDisplay(value, width - 1)}…`;
};

export const padDisplay = (
  value: string,
  width: number,
  align: Align = "left",
): string => {
  const s = truncateDisplay(String(value), width);
  const diff = width - displayWidth(s);
  if (diff <= 0) return s;
  if (align === "right") return " ".repeat(diff) + s;
  if (align === "center") {
    const left = Math.floor(diff / 2);
    return " ".repeat(left) + s + " ".repeat(diff - left);
  }
  return s + " ".repeat(diff);
};

export const wrapDisplay = (value: string, width: number): string[] => {
  const clean = sanitize(value).replace(/\s+/g, " ").trim();
  if (width <= 0) return [""];
  if (!clean) return [""];

  const lines: string[] = [];
  let current = "";
  const flush = () => {
    lines.push(current);
    current = "";
  };

  for (const word of clean.split(" ")) {
    let rest = word;
    if (current !== "" && displayWidth(current) + 1 + displayWidth(rest) <= width) {
      current = `${current} ${rest}`;
      continue;
    }
    if (current !== "") flush();
    while (displayWidth(rest) > width) {
      const cut = truncateDisplay(rest, width);
      if (cut === "") {
        lines.push(ellipsizeDisplay(rest, width));
        rest = "";
        break;
      }
      lines.push(cut);
      rest = rest.slice(cut.length);
    }
    current = rest;
  }
  if (current !== "") flush();
  return lines.length ? lines : [""];
};

const TEXTY_WIDTH = 24;


export const allocateWidths = (
  natural: number[],
  total: number,
  min = 4,
): number[] => {
  const n = natural.length;
  if (n === 0) return [];
  const budget = Math.max(n * min, Math.floor(total) - (3 * n + 1));
  const texty = natural.map((w) => w >= TEXTY_WIDTH);
  const widths = natural.map((w, i) =>
    texty[i] ? Math.min(w, 16) : Math.max(min, w),
  );
  const sum = widths.reduce((a, b) => a + b, 0);

  if (sum < budget) {
    let extra = budget - sum;
    const textIdx = natural.map((_, i) => i).filter((i) => texty[i]);
    if (textIdx.length > 0) {
      const weights = textIdx.map((i) => natural[i] ?? 1);
      const wsum = weights.reduce((a, b) => a + b, 0) || 1;
      let distributed = 0;
      textIdx.forEach((idx, k) => {
        const add = Math.floor((extra * (weights[k] ?? 1)) / wsum);
        widths[idx] = (widths[idx] ?? min) + add;
        distributed += add;
      });
      let rem = extra - distributed;
      let i = 0;
      while (rem > 0) {
        const idx = textIdx[i % textIdx.length] ?? textIdx[0]!;
        widths[idx] = (widths[idx] ?? min) + 1;
        rem -= 1;
        i += 1;
      }
    } else {
      let i = 0;
      while (extra > 0) {
        widths[i % n] = (widths[i % n] ?? min) + 1;
        extra -= 1;
        i += 1;
      }
    }
  } else if (sum > budget) {
    let over = sum - budget;
    while (over > 0) {
      let widest = -1;
      for (let i = 0; i < n; i += 1) {
        const w = widths[i] ?? 0;
        if (w > min && (widest === -1 || w > (widths[widest] ?? 0))) widest = i;
      }
      if (widest === -1) break;
      widths[widest] = (widths[widest] ?? min) - 1;
      over -= 1;
    }
  }
  return widths;
};
