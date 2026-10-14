#!/usr/bin/env python3
"""Calibrate per-language difference thresholds for the pair probe (GRU) and
DiffAlign against the SwissGov gold word labels to display nicely in TUI.

Run: python3 python/calibrate.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probationists import DEV, build_model, collate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "python" / "layerwise"
CKPT = OUT / "pairprobe_best_L14-15-16-17_sum_bce_gru_devsel.pth"
LAYERS = [14, 15, 16, 17]
LANGS = ("de", "fr", "it")
POS = 0.2


def load(role, lang):
    per = [torch.load(OUT / f"{role}_{lang}_L{l}.pt", map_location="cpu", weights_only=False) for l in LAYERS]
    out = []
    for i in range(len(per[0])):
        a = sum(per[j][i]["feats_a"].float() for j in range(len(LAYERS))) / len(LAYERS)
        b = sum(per[j][i]["feats_b"].float() for j in range(len(LAYERS))) / len(LAYERS)
        ya = np.asarray(per[0][i]["labels_a"], np.float32)[: a.shape[0]]
        yb = np.asarray(per[0][i]["labels_b"], np.float32)[: b.shape[0]]
        out.append((per[0][i]["id"], a, b, ya, yb))
    return out


@torch.no_grad()
def probe_scores(model, d_in, pairs):
    res = {}
    for i in range(0, len(pairs), 4):
        batch = pairs[i : i + 4]
        X, S, _, P = collate([(p[0], p[1].half(), p[2].half(), p[3], p[4]) for p in batch], d_in)
        out = torch.sigmoid(model(X.float().to(DEV), S.to(DEV), P.to(DEV))).cpu().numpy()
        for j, p in enumerate(batch):
            na, nb = p[1].shape[0], p[2].shape[0]
            res[p[0]] = (out[j, :na], out[j, na : na + nb], p[3], p[4])
    return res


def diffalign_scores(pairs):
    res = {}
    for p in pairs:
        an = torch.nn.functional.normalize(p[1], dim=-1)
        bn = torch.nn.functional.normalize(p[2], dim=-1)
        sim = an @ bn.t()
        res[p[0]] = ((1.0 - sim.max(dim=1).values).numpy(), (1.0 - sim.max(dim=0).values).numpy(), p[3], p[4])
    return res


def collect(per_pair):
    s, y = [], []
    for a, b, ya, yb in per_pair.values():
        s += [np.asarray(a, np.float32), np.asarray(b, np.float32)]
        y += [np.asarray(ya, np.float32), np.asarray(yb, np.float32)]
    return np.concatenate(s), np.concatenate(y)


def metrics(s, y, t):
    keep = y >= 0
    s, y = s[keep], y[keep]
    pos = y > POS
    pred = s > t
    tp = int((pred & pos).sum())
    fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum())
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    f2 = 5 * prec * rec / max(1e-9, 4 * prec + rec)
    return {
        "t": float(t), "prec": prec, "rec": rec, "f1": f1, "f2": f2,
        "tp": tp, "fp": fp, "fn": fn, "rate": (tp + fp) / max(1, int(keep.sum())),
    }


def pick(rows, key, lo_prec=0.0, max_rate=1.0):
    cands = [r for r in rows if r["prec"] >= lo_prec and r["rate"] <= max_rate]
    if not cands:
        return max(rows, key=lambda r: r["prec"])
    return max(cands, key=key)


def report(name, per, grid):
    print(f"\n== {name} ==")
    for lang in LANGS:
        s, y = collect(per[lang])
        rows = [metrics(s, y, t) for t in grid]
        cands = (
            ("F2max", pick(rows, lambda r: r["f2"])),
            ("F2@rate.25", pick(rows, lambda r: r["f2"], max_rate=0.25)),
            ("F1max", pick(rows, lambda r: r["f1"])),
            ("rec@prec.5", pick(rows, lambda r: r["rec"], lo_prec=0.5)),
        )
        for tag, m in cands:
            print(
                f"  {lang} {tag:11}: t={m['t']:.3f}  prec={m['prec']:.3f} rec={m['rec']:.3f} "
                f"f1={m['f1']:.3f} f2={m['f2']:.3f}  rate={m['rate']:.2f} tp={m['tp']} fp={m['fp']} fn={m['fn']}"
            )


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], ckpt["d_in"]).to(DEV)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"ckpt={CKPT.name} arch={ckpt['arch']} d_in={ckpt['d_in']} device={DEV}")
    probe, diff = {}, {}
    for lang in LANGS:
        pairs = load("val", lang)
        probe[lang] = probe_scores(model, ckpt["d_in"], pairs)
        diff[lang] = diffalign_scores(pairs)
    report("GRU (probe)", probe, np.arange(0.05, 0.95, 0.01))
    report("DiffAlign", diff, np.arange(0.05, 0.70, 0.005))


if __name__ == "__main__":
    main()
