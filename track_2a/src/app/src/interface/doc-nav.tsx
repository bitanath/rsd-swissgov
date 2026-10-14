import type { AppTheme } from "../libraries/hunk/ui/themes";
import {
  docsFor,
  LANG_LABEL,
  LANG_ORDER,
  SPLIT_LABEL,
  SPLIT_ORDER,
  type Doc,
  type Lang,
  type Split,
} from "../libraries/clit/dataset";

export interface NavLang {
  lang: Lang; //TODO: contains user friendly name as well as international code
  label: string;
  count: number;
  docs: Doc[];
} 

export interface NavSplit {
  split: Split;
  label: string;
  langs: NavLang[];
}

export function buildTree(): NavSplit[] {
  return SPLIT_ORDER.map((split) => ({
    split,
    label: SPLIT_LABEL[split],
    langs: LANG_ORDER.map((lang) => {
      const docs = docsFor(split, lang);
      return { lang, label: LANG_LABEL[lang], count: docs.length, docs };
    }),
  }));
}

export type NavRow =
  | { kind: "split"; id: string; label: string; split: Split; expanded: boolean }
  | { kind: "lang"; id: string; label: string; split: Split; lang: Lang; count: number; expanded: boolean }
  | { kind: "doc"; id: string; label: string; docId: string };

export const splitId = (split: Split): string => `s:${split}`;
export const langId = (split: Split, lang: Lang): string => `l:${split}:${lang}`;

export function visibleRows(
  tree: NavSplit[],
  expandedSplits: ReadonlySet<string>,
  expandedLangs: ReadonlySet<string>,
): NavRow[] {
  const rows: NavRow[] = [];
  for (const split of tree) {
    const sid = splitId(split.split);
    const splitOpen = expandedSplits.has(sid);
    rows.push({ kind: "split", id: sid, label: split.label, split: split.split, expanded: splitOpen });
    if (!splitOpen) continue;
    for (const lang of split.langs) {
      const lid = langId(split.split, lang.lang);
      const langOpen = expandedLangs.has(lid);
      rows.push({
        kind: "lang",
        id: lid,
        label: lang.label,
        split: split.split,
        lang: lang.lang,
        count: lang.count,
        expanded: langOpen,
      });
      if (!langOpen) continue;
      for (const doc of lang.docs) {
        rows.push({ kind: "doc", id: doc.id, label: doc.id, docId: doc.id });
      }
    }
  }
  return rows;
}

export function firstDocId(tree: NavSplit[]): string {
  return tree[0]?.langs[0]?.docs[0]?.id ?? "";
}

const INDENT = { split: 0, lang: 1, doc: 2 } as const;

export function DocNav({
  rows,
  selectedId,
  cursorIndex,
  width,
  theme,
  onToggleSplit,
  onToggleLang,
  onSelect,
  onCursor,
}: {
  rows: NavRow[];
  selectedId: string;
  cursorIndex: number;
  width: number;
  theme: AppTheme;
  onToggleSplit: (split: Split) => void;
  onToggleLang: (split: Split, lang: Lang) => void;
  onSelect: (docId: string) => void;
  onCursor: (index: number) => void;
}) {
  return (
    <box style={{ width: "100%", flexDirection: "column", backgroundColor: theme.panel }}>
      {rows.map((row, index) => {
        const isCursor = index === cursorIndex;
        const pad = " ".repeat(INDENT[row.kind] * 2);

        if (row.kind === "doc") {
          const selected = row.docId === selectedId;
          const bg = isCursor ? theme.selectedHunk : selected ? theme.accentMuted : theme.panel;
          const fg = selected || isCursor ? theme.text : theme.muted;
          return (
            <box
              key={row.id}
              style={{ width: "100%", height: 1, backgroundColor: bg }}
              onMouseUp={() => {
                onCursor(index);
                onSelect(row.docId);
              }}
            >
              <text fg={fg} bg={bg}>
                {`${pad}${selected ? "▸ " : "  "}${row.label}`.slice(0, Math.max(1, width - 1))}
              </text>
            </box>
          );
        }

        if (row.kind === "split") {
          const bg = isCursor ? theme.selectedHunk : theme.panelAlt;
          return (
            <box
              key={row.id}
              style={{ width: "100%", height: 1, backgroundColor: bg }}
              onMouseUp={() => {
                onCursor(index);
                onToggleSplit(row.split);
              }}
            >
              <text fg={theme.accent} bg={bg}>
                <b>{` ${row.expanded ? "▼" : "▶"} ${row.label}`.slice(0, Math.max(1, width)).toUpperCase()}</b>
              </text>
            </box>
          );
        }

        const bg = isCursor ? theme.selectedHunk : theme.panel;
        return (
          <box
            key={row.id}
            style={{ width: "100%", height: 1, backgroundColor: bg }}
            onMouseUp={() => {
              onCursor(index);
              onToggleLang(row.split, row.lang);
            }}
          >
            <text fg={theme.fileModified} bg={bg}>
              {`${pad}${row.expanded ? "▼" : "▶"} ${row.label}  (${row.count})`.slice(0, Math.max(1, width - 1))}
            </text>
          </box>
        );
      })}
    </box>
  );
}
