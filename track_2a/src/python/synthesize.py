#!/usr/bin/env python3
import argparse
import json
import re
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
WS = re.compile(r"\S+")
LOW = 0.05


def load(split, lang):
    rows = {}
    path = REPO / f"python/dataset/swissgov/{split}/{lang}.jsonl" #TODO synthesize based on already existing data
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        r["_n"] = int(r["id"].rsplit("_", 1)[1])
        rows[r["_n"]] = r
    return rows


def combine(a, b):
    """de<->fr label from the two English-relative labels."""
    if a <= LOW and b <= LOW:
        return 0.0
    if a > LOW >= b:
        return float(a)
    if b > LOW >= a:
        return float(b)
    return -1.0            # both diverge -> ambiguous


def monotonic_map(n_src, n_dst):
    """Proportional position mapping src word i -> dst word index."""
    return [min(n_dst - 1, int(i * n_dst / max(n_src, 1))) for i in range(n_src)]


def build(split, l1, l2, outdir):
    A, B = load(split, l1), load(split, l2)
    records = []
    for n in tqdm(sorted(set(A) & set(B)), desc=f"{l1}-{l2}", ncols=80):
        wa = WS.findall(A[n]["text_b"]); la = A[n]["labels_b"]
        wb = WS.findall(B[n]["text_b"]); lb = B[n]["labels_b"]
        map_ab = monotonic_map(len(wa), len(wb))

        ya = [-1.0] * len(wa)
        yb = [-1.0] * len(wb)
        for i, j in enumerate(map_ab):
            if i >= len(la) or j >= len(lb):
                continue
            v = combine(la[i], lb[j])
            ya[i] = v
            yb[j] = v

        records.append({
            "text_a": " ".join(wa), "text_b": " ".join(wb),
            "labels_a": ya, "labels_b": yb,
            "id": f"syn_{l1}-{l2}_{n}", "subset": f"{l1}-{l2}",
        })

    out = outdir / f"{l1}-{l2}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"{out}: {len(records)} pairs")


def main():
    ap = argparse.ArgumentParser(description="Synthetic target<->target RSD pairs (eval only)")
    ap.add_argument("--split", default="holdout", choices=["val", "holdout", "full"])
    ap.add_argument("--outdir", default=str(REPO / "python/dataset/swissgov/synthetic"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for l1, l2 in combinations(("de", "fr", "it"), 2):
        build(args.split, l1, l2, outdir)


if __name__ == "__main__":
    main()
