import { Modal } from "../components/modal";
import { COMMANDS } from "../libraries/commands";
import { useTheme } from "../theme/use-theme";

const KEYS: [string, string][] = [
  ["ctrl+r", "/gru — layerwise + GRU probe + English note"],
  ["ctrl+k", "/diffalign — layerwise + DiffAlign head"],
  ["ctrl+e", "/highlights — toggle the highlight display layer"],
  ["ctrl+g", "/groundtruth — highlight the gold labels (data source)"],
  ["ctrl+p", "/help — this modal"],
  ["ctrl+t", "/theme — theme chooser"],
  ["up / down", "move the sidebar cursor through headers and docs"],
  ["left / right", "collapse / expand the row under the cursor"],
  ["space / enter", "expand a group · open the highlighted document"],
  ["mouse", "click a header to collapse/expand · a doc to open"],
  ["esc", "close this modal"],
  ["ctrl+c", "/quit — exit aperdiff"],
];

function Row({
  label,
  text,
  labelColor,
  textColor,
}: {
  label: string;
  text: string;
  labelColor: string;
  textColor: string;
}) {
  return (
    <box flexDirection="row" gap={1}>
      <box width={24}>
        <text fg={labelColor}>{label}</text>
      </box>
      <text fg={textColor}>{text}</text>
    </box>
  );
}

function Section({ title, color }: { title: string; color: string }) {
  return (
    <box marginTop={1}>
      <text fg={color}>
        <b>{title}</b>
      </text>
    </box>
  );
}

export function HelpModal({ onClose }: { onClose: () => void }) {
  const theme = useTheme();

  return (
    <Modal title="help" width={68} height={22} onClose={onClose}>
      <scrollbox flexGrow={1} minHeight={0}>
        <text fg={theme.colors.foreground}>
          <b>aperdiff</b> — a Hunk-style review surface for semantic language diffing.
        </text>

        <Section title="commands" color={theme.colors.primary} />
        {COMMANDS.map((command) => (
          <Row
            key={command.name}
            label={`/${command.name}`}
            text={command.keys ? `${command.description}   (${command.keys})` : command.description}
            labelColor={theme.colors.accent}
            textColor={theme.colors.mutedForeground}
          />
        ))}

        <Section title="keys" color={theme.colors.primary} />
        {KEYS.map(([label, text]) => (
          <Row
            key={label}
            label={label}
            text={text}
            labelColor={theme.colors.accent}
            textColor={theme.colors.mutedForeground}
          />
        ))}
      </scrollbox>
    </Modal>
  );
}
