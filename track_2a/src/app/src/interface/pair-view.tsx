import { Alert } from "../components/alert";
import type { AppTheme } from "../libraries/hunk/ui/themes";
import { words, type Doc } from "../libraries/clit/dataset";
import { HEAD_LABEL, headMask, PROV_GRU, type Combined, type Head, type Signal } from "../libraries/clit/analyze";

interface Word {
  text: string;
  idx: number;
}

type HighlightMode = "gold" | "score"; //HACK: We either show the ground truth or the model scored values, no in-betweeners

const LINE_HEAT_RATIO = 0.15;

const AGENT_ICON = "🤖";
const AGENT_ICON_FALLBACK = "✏️";

function agentIcon(): string {
  const term = process.env.TERM ?? "";
  const noEmoji = process.env.APERDIFF_NO_EMOJI === "1" || /^(linux|dumb)$/.test(term);
  return noEmoji ? AGENT_ICON_FALLBACK : AGENT_ICON;
}

function layout(tokens: string[], colW: number): Word[][] {
  const lines: Word[][] = [];
  let cur: Word[] = [];
  let curLen = 0;
  tokens.forEach((text, idx) => {
    const add = text.length + (cur.length > 0 ? 1 : 0);
    if (cur.length > 0 && curLen + add > colW) {
      lines.push(cur);
      cur = [];
      curLen = 0;
    }
    cur.push({ text, idx });
    curLen += text.length + (cur.length > 1 ? 1 : 0);
  });
  if (cur.length > 0) lines.push(cur);
  return lines;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  const v = Number.parseInt(h.length === 3 ? h.replace(/(.)/g, "$1$1") : h, 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function mix(a: string, b: string, t: number): string {
  const [r1, g1, b1] = hexToRgb(a);
  const [r2, g2, b2] = hexToRgb(b);
  const c = (x: number, y: number) => Math.round(x + (y - x) * t);
  return `#${[c(r1, r2), c(g1, g2), c(b1, b2)]
    .map((n) => n.toString(16).padStart(2, "0"))
    .join("")}`;
}

function heatColor(theme: AppTheme, t: number, hot: string): string {
  return mix(theme.panel, hot, Math.min(1, Math.max(0, t)) * 0.85);
}

function maxOf(values: ArrayLike<number> | null | undefined): number {
  if (!values) return 0;
  let max = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] ?? 0;
    if (v > max) max = v;
  }
  return max;
}

function Column({
  label,
  tokens,
  values,
  prov,
  mask = 3,
  palette = "provenance",
  show,
  mode,
  threshold,
  peak,
  width,
  theme,
}: {
  label: string;
  tokens: string[];
  values: ArrayLike<number> | null;
  prov?: Uint8Array | null;
  mask?: number;
  palette?: "provenance" | "gru" | "diff";
  show: boolean;
  mode: HighlightMode;
  threshold: number;
  peak: number;
  width: number;
  theme: AppTheme;
}) {
  const lines = layout(tokens, Math.max(8, width));
  const hot = mode === "score" ? theme.removedSignColor : theme.fileModified;
  const span = Math.max(1e-6, peak - threshold);

  
  const wordStyle = (idx: number): { bg: string | null; intensity: number } => {
    if (!show) return { bg: null, intensity: 0 };
    if (prov) {
      const p = (prov[idx] ?? 0) & mask;
      if (!p) return { bg: null, intensity: 0 };
      //FIXME: provenance colours are a rough legend: gru = amber, diff = blue, both = red.
      const tint =
        palette === "provenance"
          ? p === 3
            ? theme.removedSignColor
            : p === PROV_GRU
              ? theme.addedSignColor
              : theme.fileModified
          : palette === "gru"
            ? theme.addedSignColor
            : theme.fileModified;
      return { bg: heatColor(theme, 0.85, tint), intensity: 1 };
    }
    if (!values) return { bg: null, intensity: 0 };
    const value = values[idx] ?? 0;
    if (!(value > threshold)) return { bg: null, intensity: 0 };
    const intensity = 0.45 + 0.55 * Math.min(1, (value - threshold) / span); //HACK: per word heatmap intensity (if it counts as a mismatch) and 0..1 intensity.
    return { bg: heatColor(theme, intensity, hot), intensity };
  };

  return (
    <box style={{ width, flexDirection: "column", backgroundColor: theme.panel }}>
      <text fg={theme.accent} bg={theme.panel}>
        <b>{label}</b>
      </text>
      {lines.map((line, li) => {
        const styles = line.map((w) => wordStyle(w.idx));
        const strongest = styles.reduce((max, s) => Math.max(max, s.intensity), 0);
        const lineBg = strongest > 0 ? heatColor(theme, strongest * LINE_HEAT_RATIO, hot) : theme.panel;
        return (
          <box key={li} style={{ width: "100%", backgroundColor: lineBg }}>
            <text bg={lineBg} fg={theme.text}>
              {line.map((w, wi) => (
                <span key={wi} fg={theme.text} bg={styles[wi]?.bg ?? lineBg}>
                  {wi > 0 ? " " : ""}
                  {w.text}
                </span>
              ))}
            </text>
          </box>
        );
      })}
    </box>
  );
}

export function PairView({
  doc,
  width,
  theme,
  combined,
  head,
  signal,
  showHighlights,
  showGold,
}: {
  doc: Doc;
  width: number;
  theme: AppTheme;
  combined?: Combined | null;
  head: Head | null;
  signal: Signal | null;
  showHighlights: boolean;
  showGold: boolean;
}) {
  const colW = Math.max(12, Math.floor((width - 1) / 2));
  const a = words(doc.text_a);
  const b = words(doc.text_b);

  const model = !showGold && combined != null && head != null;
  const mask = headMask(head);
  const palette = head === "combined" ? "provenance" : head === "gru" ? "gru" : "diff";
  const goldPeak = Math.max(1, maxOf(doc.labels_a), maxOf(doc.labels_b));

  const sourceLabel = showGold ? "ground truth" : head ? HEAD_LABEL[head] : "model";
  const modeLabel = !showHighlights ? "off" : sourceLabel;
  const verdict = signal ? `  ·  ${signal}` : "";
  const legend =
    head === "combined"
      ? [
          <span key="lg1">{"  ·  "}</span>,
          <span key="lg2" fg={theme.addedSignColor}>gru</span>,
          <span key="lg3">{" · "}</span>,
          <span key="lg4" fg={theme.fileModified}>diff</span>,
          <span key="lg5">{" · "}</span>,
          <span key="lg6" fg={theme.removedSignColor}>both</span>,
        ]
      : [];

  return (
    <box style={{ width: "100%", flexDirection: "column", backgroundColor: theme.background }}>
      <text fg={theme.muted} bg={theme.background}>
        {[
          <span key="pre">{`${doc.id}  ·  ${doc.split}  ·  ${doc.lang.toUpperCase()}  ·  highlight: `}</span>,
          <span key="mode">{modeLabel}</span>,
          ...legend,
          <span key="verdict">{verdict}</span>,
        ]}
      </text>
      <box style={{ width: "100%", flexDirection: "row", backgroundColor: theme.panel }}>
        <Column
          label="EN · text_a"
          tokens={a}
          values={showGold ? doc.labels_a : null}
          prov={model ? combined.provA : null}
          mask={mask}
          palette={palette}
          show={showHighlights}
          mode="gold"
          threshold={0}
          peak={goldPeak}
          width={colW}
          theme={theme}
        />
        <box style={{ width: 1, backgroundColor: theme.background }} />
        <Column
          label={`${doc.lang.toUpperCase()} · text_b`}
          tokens={b}
          values={showGold ? doc.labels_b : null}
          prov={model ? combined.provB : null}
          mask={mask}
          palette={palette}
          show={showHighlights}
          mode="gold"
          threshold={0}
          peak={goldPeak}
          width={Math.max(12, width - colW - 1)}
          theme={theme}
        />
      </box>
    </box>
  );
}

export function NotePanel({
  note,
  verdict,
  head,
  running,
  width,
  height = 7,
  icon = agentIcon(),
}: {
  note: string;
  verdict?: string | null;
  head?: Head | null;
  running: boolean;
  width: number;
  theme: AppTheme;
  height?: number;
  icon?: string;
}) {
  const contentWidth = Math.max(4, width - 4);
  const maxLines = Math.max(1, height - 3);
  const source =
    verdict ??
    (note.length > 0
      ? note
      : running
        ? "…"
        : "Press ^R for pairprobe (supervised), or ^K for diffalign (unsupervised).");

  const lines: string[] = [];
  let cur = "";
  for (const w of words(source)) {
    if (cur.length > 0 && cur.length + 1 + w.length > contentWidth) {
      lines.push(cur);
      cur = "";
    }
    cur = cur.length > 0 ? `${cur} ${w}` : w;
  }
  if (cur.length > 0) lines.push(cur);

  const shown = lines.slice(0, maxLines);
  while (shown.length < maxLines) shown.push("");

  return (
    <box style={{ width: "100%", flexShrink: 0 }}>
      <Alert
        variant="info"
        icon={icon}
        title={verdict ? "agent note · ground truth" : head ? `agent note · ${HEAD_LABEL[head]}` : "agent note"}
        message={shown.join("\n")}
      />
    </box>
  );
}
