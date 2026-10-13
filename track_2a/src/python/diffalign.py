#!/usr/bin/env python3
"""
Unsupervised DiffAlign on frozen Apertus features. Replicating (Wastl et al 2026).
diff(token) = 1 - max cosine to any token of the other text (Vamvas & Sennrich 2023).

Usage: diffalign.py [layers e.g. 15 | 14,15,16,17]
"""

import sys, os, numpy as np, torch
from pathlib import Path
from scipy.stats import spearmanr

OUT = Path(os.environ.get("LW", "./aperdiff/python/layerwise"))
LAYERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "14,15,16,17").split(",")]
CEIL = {"de": 0.561, "fr": 0.436, "it": 0.651}
ROLE = os.environ.get("ROLE", "val")


def load(role, lang, layers):
    first = layers[0]
    p = OUT / f"{role}_{lang}_L{first}.pt"
    per = [torch.load(OUT / f"{role}_{lang}_L{l}.pt", map_location="cpu", weights_only=False) for l in layers]
    out = []
    for i in range(len(per[0])):
        e = per[0][i]
        A = sum(per[j][i]["feats_a"].float() for j in range(len(layers))) / len(layers)
        B = sum(per[j][i]["feats_b"].float() for j in range(len(layers))) / len(layers)
        ya = np.asarray(e["labels_a"], np.float32)[:A.shape[0]]
        yb = np.asarray(e["labels_b"], np.float32)[:B.shape[0]]
        out.append((A, B, ya, yb))
    del per
    return out


@torch.no_grad()
def diff_scores(A, B):
    An = torch.nn.functional.normalize(A, dim=-1)
    Bn = torch.nn.functional.normalize(B, dim=-1)
    sim = An @ Bn.t()                       # (na, nb) cosine
    da = 1.0 - sim.max(dim=1).values        # each A token vs best B
    db = 1.0 - sim.max(dim=0).values        # each B token vs best A
    return da.numpy(), db.numpy()


def main():
    for mode, layers in [("L15", [15]), ("L14-17 sum", LAYERS)]:
        print(f"\n=== DiffAlign · Apertus {mode} ===", flush=True)
        vals = {}
        for lang in ("de", "fr", "it"):
            P, G = [], []
            for A, B, ya, yb in load(ROLE, lang, layers):
                da, db = diff_scores(A, B)
                P += [da, db]; G += [ya, yb]
            p = np.concatenate(P); g = np.concatenate(G)
            m = g >= 0
            r = spearmanr(p[m], g[m]).statistic
            vals[lang] = r
            print(f"  {lang:>3}: rho={r:.4f}  ({r/CEIL[lang]*100:5.1f}%ceil)", flush=True)
        print("  mean = %.4f" % np.mean(list(vals.values())), flush=True)


if __name__ == "__main__":
    main()
