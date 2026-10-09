//TODO replace with actual tokenized logic
import { diffWordsWithSpace } from "diff";

export type WordRange = readonly [start: number, end: number];

export interface WordDiffRanges {
  left: WordRange[];
  right: WordRange[];
}

function mergeRanges(ranges: WordRange[]): WordRange[] {
  if (ranges.length <= 1) return ranges;
  const sorted = [...ranges].sort((a, b) => a[0] - b[0]);
  const merged: WordRange[] = [sorted[0]!];
  for (let i = 1; i < sorted.length; i += 1) {
    const [start, end] = sorted[i]!;
    const last = merged[merged.length - 1]!;
    if (start <= last[1]) {
      if (end > last[1]) merged[merged.length - 1] = [last[0], end];
    } else {
      merged.push([start, end]);
    }
  }
  return merged;
}

export function wordDiffRanges(left: string, right: string): WordDiffRanges {
  if (left === right) {
    return { left: [], right: [] };
  }

  const leftRanges: WordRange[] = [];
  const rightRanges: WordRange[] = [];
  let leftOffset = 0;
  let rightOffset = 0;

  for (const part of diffWordsWithSpace(left, right)) {
    const length = part.value.length;
    if (part.removed) {
      if (length > 0) leftRanges.push([leftOffset, leftOffset + length]);
      leftOffset += length;
    } else if (part.added) {
      if (length > 0) rightRanges.push([rightOffset, rightOffset + length]);
      rightOffset += length;
    } else {
      leftOffset += length;
      rightOffset += length;
    }
  }

  return { left: mergeRanges(leftRanges), right: mergeRanges(rightRanges) };
}
