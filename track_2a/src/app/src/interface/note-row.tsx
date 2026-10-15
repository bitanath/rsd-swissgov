//TODO Per-file note row: the "agent message body" slot. Plain wrapped text inside a Hunk card
import { fitText, measureTextWidth, padText, wrapText } from "../libraries/hunk/ui/lib/text";
import type { DiffRow } from "../libraries/hunk/ui/diff/diffRowModel";
import type { AppTheme } from "../libraries/hunk/ui/themes";

type NoteRow = Extract<DiffRow, { type: "note" }>;

export function NoteRowView({
  row,
  width,
  theme,
}: {
  row: NoteRow;
  width: number;
  theme: AppTheme;
}) {
  const safeWidth = Math.max(4, width);
  const contentWidth = Math.max(1, safeWidth - 2);
  const hasBody = row.text.trim().length > 0;
  const bodyLines = hasBody
    ? row.text.split("\n").flatMap((line) => wrapText(line, contentWidth))
    : ["No note yet"];

  const titleText = fitText(` ${row.title} `, Math.max(0, safeWidth - 3));
  const titleWidth = measureTextWidth(titleText);
  const topSuffix = "─".repeat(Math.max(0, safeWidth - 3 - titleWidth));

  return (
    <box style={{ width: "100%", flexDirection: "column", backgroundColor: theme.panel }}>
      <text bg={theme.panel}>
        <span fg={theme.noteBorder}>╭─</span>
        <span fg={theme.noteTitleText}>{titleText}</span>
        <span fg={theme.noteBorder}>{topSuffix}╮</span>
      </text>

      {bodyLines.map((line, index) => (
        <text key={index} bg={theme.panel}>
          <span fg={theme.noteBorder}>│</span>
          <span fg={hasBody ? theme.text : theme.muted}>{padText(line, contentWidth)}</span>
          <span fg={theme.noteBorder}>│</span>
        </text>
      ))}

      <text bg={theme.panel}>
        <span fg={theme.noteBorder}>{`╰${"─".repeat(Math.max(0, safeWidth - 2))}╯`}</span>
      </text>
    </box>
  );
}
