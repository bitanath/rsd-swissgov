import { HEAD_LABEL } from "./analyze";

export type ToastVariant = "success" | "error" | "warning" | "info";
export type ModalKind = "help" | "theme";

export interface CommandContext {
  quit: () => void;
  openModal: (modal: ModalKind) => void;
  toast: (message: string, variant?: ToastVariant) => void;
  runGru: () => void;
  runDiffalign: () => void;
  runCombined: () => void;
  toggleHighlights: () => void;
  toggleGold: () => void;
}

export interface CommandDef {
  name: string;
  label: string;
  description: string;
  keys?: string;
}

export const COMMANDS: CommandDef[] = [
  { name: "supervisedalign", label: HEAD_LABEL.gru, description: "layerwise + pairprobe (supervised) + English note", keys: "ctrl+r" },
  { name: "diffalign", label: HEAD_LABEL.diffalign, description: "layerwise + diffalign (unsupervised) head + English note", keys: "ctrl+k" },
  { name: "combined", label: HEAD_LABEL.combined, description: "both heads: union of differences + grounded note", keys: "ctrl+d" },
  { name: "highlights", label: "highlights", description: "toggle the highlight display layer", keys: "ctrl+e" },
  { name: "groundtruth", label: "groundtruth", description: "highlight the gold labels (data source)", keys: "ctrl+g" },
  { name: "theme", label: "theme", description: "open the theme chooser", keys: "ctrl+t" },
  { name: "help", label: "help", description: "list commands and shortcut keys", keys: "ctrl+p" },
  { name: "quit", label: "quit", description: "exit aperdiff", keys: "ctrl+c" },
];

export function runCommand(input: string, ctx: CommandContext): void {
  const trimmed = input.trim();
  if (trimmed.length === 0) return;
  if (!trimmed.startsWith("/")) return;

  const [raw, ...rest] = trimmed.split(/\s+/);
  const name = (raw ?? "").slice(1).toLowerCase();
  const arg = rest.join(" ");

  switch (name) {
    case "supervisedalign":
    case "gru":
      ctx.runGru();
      return;
    case "diffalign":
      ctx.runDiffalign();
      return;
    case "combined":
      ctx.runCombined();
      return;
    case "highlights":
      ctx.toggleHighlights();
      return;
    case "groundtruth":
    case "gold":
      ctx.toggleGold();
      return;
    case "theme":
      ctx.openModal("theme");
      return;
    case "help":
      ctx.openModal("help");
      return;
    case "quit":
      ctx.quit();
      return;
    default:
      ctx.toast(`unknown /${name}${arg ? ` ${arg}` : ""} — try /help`, "warning");
  }
}
