import { Modal } from "../components/modal";
import { COMMANDS } from "../libraries/commands";
import { useTheme } from "../theme/use-theme";

const KEYS: [string, string][] = [
  ["ctrl+t", "theme chooser"],
  ["ctrl+r", "re-render the diff"],
  ["mouse drag", "select text · auto-copies"],
  ["enter", "run the highlighted command"],
  ["esc", "close this modal"],
  ["ctrl+c", "quit"],
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
            text={command.description}
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
