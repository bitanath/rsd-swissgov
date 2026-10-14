export type LayoutMode = "split" | "unified";
export type ToastVariant = "success" | "error" | "warning" | "info";
export type ModalKind = "help" | "theme";

export interface CommandContext {
  layout: LayoutMode;
  setLayout: (mode: LayoutMode) => void;
  quit: () => void;
  openModal: (modal: ModalKind) => void;
  toast: (message: string, variant?: ToastVariant) => void;
  runGru: () => void;
  runDiffalign: () => void;
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
  { name: "gru", label: "gru", description: "layerwise + GRU probe + English note", keys: "ctrl+r" },
  { name: "diffalign", label: "diffalign", description: "layerwise + DiffAlign head", keys: "ctrl+k" },
  { name: "highlights", label: "highlights", description: "toggle the highlight display layer", keys: "ctrl+e" },
  { name: "groundtruth", label: "groundtruth", description: "highlight the gold labels (data source)", keys: "ctrl+g" },
  { name: "theme", label: "theme", description: "open the theme chooser", keys: "ctrl+t" },
  { name: "help", label: "help", description: "list commands and shortcut keys", keys: "ctrl+p" },
  { name: "quit", label: "quit", description: "exit aperdiff", keys: "ctrl+c" },
  { name: "split", label: "split", description: "side-by-side layout" },
  { name: "unified", label: "unified", description: "stacked unified layout" },
];

export function runCommand(input: string, ctx: CommandContext): void {
  const trimmed = input.trim();
  if (trimmed.length === 0) return;
  if (!trimmed.startsWith("/")) return;

  const [raw, ...rest] = trimmed.split(/\s+/);
  const name = (raw ?? "").slice(1).toLowerCase();
  const arg = rest.join(" ");

  switch (name) {
    case "gru":
      ctx.runGru();
      return;
    case "diffalign":
      ctx.runDiffalign();
      return;
    case "highlights":
      ctx.toggleHighlights();
      return;
    case "groundtruth":
    case "gold":
      ctx.toggleGold();
      return;
    case "split":
      ctx.setLayout("split");
      ctx.toast("layout: split", "success");
      return;
    case "unified":
      ctx.setLayout("unified");
      ctx.toast("layout: unified", "success");
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
