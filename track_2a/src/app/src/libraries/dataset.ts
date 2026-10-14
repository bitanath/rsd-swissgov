import raw from "../data/swissgov.json"; //NOTE ground truths

export type Lang = "de" | "fr" | "it";
export type Split = "test" | "dev";

export interface Doc {
  id: string;
  split: Split;
  lang: Lang;
  text_a: string;
  text_b: string;
  labels_a: number[];
  labels_b: number[];
}

export const DOCS = raw as unknown as Doc[];

export const SPLIT_ORDER: Split[] = ["test", "dev"];
export const LANG_ORDER: Lang[] = ["de", "fr", "it"];
export const SPLIT_LABEL: Record<Split, string> = { test: "Test", dev: "Dev" };
export const LANG_LABEL: Record<Lang, string> = { de: "DE (German)", fr: "FR (French)", it: "IT (Italian)" };

const WORD_RE = /[^\t\n\v\f\r ]+/g; //FIXME: Match the zig word boundaries exactly

export function words(text: string): string[] {
  return text.match(WORD_RE) ?? [];
}

export function wordCount(text: string): number {
  return words(text).length;
}

export function docsFor(split: Split, lang: Lang): Doc[] {
  return DOCS.filter((d) => d.split === split && d.lang === lang).sort(
    (x, y) => idNum(x.id) - idNum(y.id),
  );
}

function idNum(id: string): number {
  return Number.parseInt(id.slice(id.lastIndexOf("_") + 1), 10);
}

export const ALL_DOC_IDS = DOCS.map((d) => d.id);

export function findDoc(id: string): Doc | undefined {
  return DOCS.find((d) => d.id === id);
}
