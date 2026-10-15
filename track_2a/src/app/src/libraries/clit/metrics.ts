// Token-level rank correlations against gold, matching the SwissGov-RSD judging
// harness (ZurichNLP/SwissGov-RSD): labels pooled over both texts, gold -1
// dropped, a short prediction zero-padded / a long one truncated to gold length.
//
// Used by the headless `results`/`predict` tables to expose the paper's metrics
// (token-level Spearman rho and Kendall tau-b) per document.

/** Average ranks (1-based); tied values share their mean rank. */
function averageRanks(values: number[]): number[] {
  const order = values.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const ranks = new Array<number>(values.length);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1]![0] === order[i]![0]) j += 1;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k += 1) ranks[order[k]![1]] = avg;
    i = j + 1;
  }
  return ranks;
}

function pearson(a: number[], b: number[]): number {
  const n = a.length;
  if (n < 2) return Number.NaN;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i += 1) {
    ma += a[i]!;
    mb += b[i]!;
  }
  ma /= n;
  mb /= n;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < n; i += 1) {
    const da = a[i]! - ma;
    const db = b[i]! - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  const den = Math.sqrt(saa * sbb);
  return den > 0 ? sab / den : Number.NaN;
}

/** Spearman rho = Pearson correlation of average ranks. NaN when undefined. */
export function spearman(pred: number[], gold: number[]): number {
  if (pred.length !== gold.length || pred.length < 3) return Number.NaN;
  return pearson(averageRanks(pred), averageRanks(gold));
}

/** Kendall tau-b with tie correction. NaN when undefined. */
export function kendallTauB(pred: number[], gold: number[]): number {
  const n = pred.length;
  if (pred.length !== gold.length || n < 3) return Number.NaN;

  // Coordinate-compress gold so counting uses a Fenwick tree over <= |gold| buckets.
  const distinct = Array.from(new Set(gold)).sort((a, b) => a - b);
  const goldRank = new Map<number, number>();
  distinct.forEach((v, i) => goldRank.set(v, i + 1));
  const K = distinct.length;
  const bit = new Array<number>(K + 1).fill(0);
  const add = (i: number): void => {
    for (; i <= K; i += i & -i) bit[i]! += 1;
  };
  const prefix = (i: number): number => {
    let s = 0;
    for (; i > 0; i -= i & -i) s += bit[i]!;
    return s;
  };

  // Process predictions in ascending order; equal-prediction groups are tied in
  // x, so they never contribute concordant/discordant pairs and are added after.
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => pred[a]! - pred[b]!);

  let concordant = 0;
  let discordant = 0;
  let seen = 0;
  let n1 = 0; // pairs tied in pred
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && pred[order[j + 1]!]! === pred[order[i]!]!) j += 1;
    for (let k = i; k <= j; k += 1) {
      const r = goldRank.get(gold[order[k]!]!)!;
      concordant += prefix(r - 1); // earlier, smaller gold
      discordant += seen - prefix(r); // earlier, larger gold
    }
    for (let k = i; k <= j; k += 1) add(goldRank.get(gold[order[k]!]!)!);
    const t = j - i + 1;
    n1 += (t * (t - 1)) / 2;
    seen += t;
    i = j + 1;
  }

  // n2 = pairs tied in gold.
  const counts = new Map<number, number>();
  for (const g of gold) counts.set(g, (counts.get(g) ?? 0) + 1);
  let n2 = 0;
  for (const c of counts.values()) n2 += (c * (c - 1)) / 2;

  const n0 = (n * (n - 1)) / 2;
  const den = Math.sqrt((n0 - n1) * (n0 - n2));
  return den > 0 ? (concordant - discordant) / den : Number.NaN;
}

/** Zero-pad a short prediction / truncate a long one, like the judge harness. */
export function align(pred: number[], gold: number[]): number[] {
  const out = pred.slice(0, gold.length);
  while (out.length < gold.length) out.push(0);
  return out;
}

export interface DocMetric {
  spearman: number;
  kendall: number;
}

/**
 * One document pair: drop text b when its gold is entirely -1 (unlabelled side),
 * then drop every remaining gold -1 position before scoring.
 */
export function docMetric(
  predA: number[],
  predB: number[],
  goldA: number[],
  goldB: number[],
): DocMetric {
  const pred = align(predA, goldA);
  const gold = goldA.slice();
  if (goldB.some((x) => x !== -1)) {
    pred.push(...align(predB, goldB));
    gold.push(...goldB);
  }
  const fp: number[] = [];
  const fg: number[] = [];
  for (let i = 0; i < gold.length; i += 1) {
    if (gold[i] !== -1) {
      fp.push(pred[i]!);
      fg.push(gold[i]!);
    }
  }
  return { spearman: spearman(fp, fg), kendall: kendallTauB(fp, fg) };
}
