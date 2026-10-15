import { useTheme } from "../theme/use-theme";

export type AutocompleteOption = { name: string; label: string; description: string };

export function Autocomplete({
  options,
  selected,
  maxHeight = 10,
}: {
  options: AutocompleteOption[];
  selected: number;
  maxHeight?: number;
}) {
  const theme = useTheme();
  const total = options.length;
  const size = Math.min(maxHeight, Math.max(1, total));
  const start = selected >= size ? selected - size + 1 : 0;
  const visible = options.slice(start, start + size);
  const hiddenAbove = start;
  const hiddenBelow = Math.max(0, total - start - visible.length);

  return (
    <box borderStyle="rounded" borderColor={theme.colors.border} flexDirection="column" flexShrink={0}>
      {hiddenAbove > 0 && (
        <box paddingLeft={1} paddingRight={1}>
          <text fg={theme.colors.mutedForeground}>{`↑ ${hiddenAbove} more`}</text>
        </box>
      )}
      {visible.length === 0 ? (
        <box paddingLeft={1} paddingRight={1}>
          <text fg={theme.colors.mutedForeground}>No matching items</text>
        </box>
      ) : (
        visible.map((option, index) => {
          const optionIndex = start + index;
          const active = optionIndex === selected;
          const fg = active ? theme.colors.background : theme.colors.foreground;
          const dim = active ? theme.colors.background : theme.colors.mutedForeground;
          return (
            <box
              key={option.name}
              flexDirection="row"
              paddingLeft={1}
              paddingRight={1}
              backgroundColor={active ? theme.colors.primary : undefined}
            >
              <text fg={fg} flexShrink={0}>{`/${option.name}`}</text>
              <text fg={dim} wrapMode="none">{`  ${option.description}`}</text>
            </box>
          );
        })
      )}
      {hiddenBelow > 0 && (
        <box paddingLeft={1} paddingRight={1}>
          <text fg={theme.colors.mutedForeground}>{`↓ ${hiddenBelow} more`}</text>
        </box>
      )}
    </box>
  );
}
