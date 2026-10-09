export type LayoutMode = "split" | "unified";
export type ToastVariant = "success" | "error" | "warning" | "info";
export type ModalKind = "help" | "theme";

export interface CommandContext {
  layout: LayoutMode;
  setLayout: (mode: LayoutMode) => void;
  quit: () => void;
  openModal: (modal: ModalKind) => void;
  toast: (message: string, variant?: ToastVariant) => void;
}

export interface CommandDef {
  name: string;
  label: string;
  description: string;
}

export const COMMANDS: CommandDef[] = [
  { name: "split", label: "split", description: "side-by-side layout" },
  { name: "unified", label: "unified", description: "stacked unified layout" },
  { name: "theme", label: "theme", description: "open the theme chooser" },
  { name: "help", label: "help", description: "list commands and keys" },
  { name: "quit", label: "quit", description: "exit aperdiff" },
]; //TODO remove split vs unified, or merge it into single command

export function runCommand(input: string, ctx: CommandContext): void {
  const trimmed = input.trim();
  if (trimmed.length === 0) {
    return;
  }

  if (!trimmed.startsWith("/")) { return}

  const [raw, ...rest] = trimmed.split(/\s+/);
  const name = (raw ?? "").slice(1).toLowerCase();
  const arg = rest.join(" ");

  switch (name) {
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
