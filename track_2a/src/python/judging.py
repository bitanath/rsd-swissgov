#!/usr/bin/env python3
"""Replicate the official SwissGov-RSD judging harness, reusing frozen-Apertus features.

    layerwise/train_{lang}_L14..17.pt   -> dev  split (168/lang)
    layerwise/val_{lang}_L14..17.pt     -> test split ( 56/lang)

Systems:
  difffalign  unsupervised DiffAlign on frozen Apertus (no training)
  probe       trained pair probe checkpoint (.pth)

Usage:
  python judging.py --system difffalign
  python judging.py --system probe --tag devsel --arch gru
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
while not (REPO / "core").exists():
    REPO = REPO.parent
OUT = REPO / "python" / "layerwise"
DATA = REPO / "python" / "dataset" / "swissgov"
JUDGING = REPO / "python" / "dataset" / "judging"
os.environ.setdefault("LW", str(OUT))          #FIXME keep probationists' OUT in sync
sys.path.insert(0, str(REPO / "python"))

import probationists as P

LANGS = ("de", "fr", "it")
WS = re.compile(r"\S+")

SPLIT_DIR = {"dev": "val", "test": "holdout", "full": "full"} #FIXME match directory names in dataset/swissgov

def load_role(role, lang, layers):
    """Return {id: {tokens_a, tokens_b, labels_a, labels_b, per_layer_feats}}."""
    files = [OUT / f"{role}_{lang}_L{l}.pt" for l in layers]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"missing feature cache(s): {missing}")
    per = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    docs = {}
    for i in range(len(per[0])):
        e0 = per[0][i]
        docs[e0["id"]] = {
            "id": e0["id"],
            "tokens_a": list(e0["tokens_a"]),
            "tokens_b": list(e0["tokens_b"]),
            "labels_a": [float(x) for x in e0["labels_a"]],
            "labels_b": [float(x) for x in e0["labels_b"]],
            "feats_a": [per[j][i]["feats_a"].float() for j in range(len(layers))],
            "feats_b": [per[j][i]["feats_b"].float() for j in range(len(layers))],
            "_dev": role == "train",
        }
    del per
    return docs


def load_full(lang, layers):
    """Merge dev (train) + test (val) captures into full-set order by numeric id."""
    dev = load_role("train", lang, layers)
    test = load_role("val", lang, layers)
    overlap = set(dev) & set(test)
    if overlap:
        raise RuntimeError(f"{lang}: dev/test id overlap {sorted(overlap)[:5]}")
    merged = {}
    merged.update(dev)
    merged.update(test)
    ordered = sorted(merged.values(), key=lambda d: int(d["id"].rsplit("_", 1)[1]))
    return ordered


def fuse(feats, layers, fusion):
    if fusion == "sum":
        return sum(feats) / len(layers)
    return torch.cat(feats, dim=-1)


@torch.no_grad()
def predict_difffalign(docs, layers, fusion):
    for d in docs:
        A = fuse(d["feats_a"], layers, fusion)
        B = fuse(d["feats_b"], layers, fusion)
        An = torch.nn.functional.normalize(A, dim=-1)
        Bn = torch.nn.functional.normalize(B, dim=-1)
        sim = An @ Bn.t()
        d["pred_a"] = (1.0 - sim.max(dim=1).values).numpy().tolist()
        d["pred_b"] = (1.0 - sim.max(dim=0).values).numpy().tolist()


@torch.no_grad()
def predict_probe(docs, ckpt_path, layers, fusion):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    model = P.build_model(arch, ckpt["d_in"]).to(P.DEV)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  loaded {ckpt_path.name} (arch={arch}, seed={ckpt.get('seed')}, "
          f"epoch={ckpt.get('epoch')}, val_macro={ckpt.get('val_macro')})", flush=True)
    for d in docs:
        A = fuse(d["feats_a"], layers, fusion)
        B = fuse(d["feats_b"], layers, fusion)
        ya = np.asarray(d["labels_a"], np.float32)
        yb = np.asarray(d["labels_b"], np.float32)
        na, nb = A.shape[0], B.shape[0]
        X, S, Y, Pm = P.collate([(d["id"], A.half(), B.half(), ya[:na], yb[:nb])], ckpt["d_in"])
        out = torch.sigmoid(model(X.float().to(P.DEV), S.to(P.DEV), Pm.to(P.DEV))).cpu()
        d["pred_a"] = out[0, :na].tolist()
        d["pred_b"] = out[0, na:na + nb].tolist()


def write_predictions(docs, prefix):
    JUDGING.mkdir(parents=True, exist_ok=True)
    paths = {}
    for lang in LANGS:
        rows = [d for d in docs if d["id"].startswith(f"admin_{lang}_")]
        p = JUDGING / f"{prefix}{lang}.jsonl"
        with open(p, "w") as f:
            for d in rows:
                f.write(json.dumps({
                    "id": d["id"],
                    "text_a": " ".join(d["tokens_a"]),
                    "text_b": " ".join(d["tokens_b"]),
                    "labels_a": d["pred_a"],
                    "labels_b": d["pred_b"],
                }) + "\n")
        paths[lang] = p
    return paths


def _align(pred, gold):
    """Match the harness: zero-pad a short prediction, truncate a long one."""
    pred = list(pred)
    if len(pred) < len(gold):
        pred = pred + [0.0] * (len(gold) - len(pred))
    elif len(pred) > len(gold):
        pred = pred[:len(gold)]
    return pred


def gold_rows(lang, split):
    with open(DATA / SPLIT_DIR[split] / f"{lang}.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


def score_split(docs, lang, split):
    """Global Spearman + Kendall tau-b via nlpstats, gold -1 dropped (judge metric)."""
    from nlpstats.correlations import correlate
    pred_by_id = {d["id"]: d for d in docs}
    pred_labels, gold_labels = [], []
    for g in gold_rows(lang, split):
        d = pred_by_id.get(g["id"])
        if d is None:
            continue
        ga = [float(x) for x in g["labels_a"]]
        pa = _align(d["pred_a"], ga)
        pred_labels.extend(pa); gold_labels.extend(ga)
        gb = [float(x) for x in g["labels_b"]]
        if not all(x == -1 for x in gb):
            pb = _align(d["pred_b"], gb)
            pred_labels.extend(pb); gold_labels.extend(gb)
    p = np.asarray(pred_labels, np.float32)
    y = np.asarray(gold_labels, np.float32)
    m = y != -1
    p, y = p[m], y[m]
    X = np.expand_dims(p, 0); Y = np.expand_dims(y, 0)
    sp = float(np.asarray(correlate(X, Y, level="global", coefficient="spearman")).ravel()[0])
    kd = float(np.asarray(correlate(X, Y, level="global", coefficient="kendall")).ravel()[0])
    return sp, kd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="difffalign", choices=["difffalign", "probe", "both"])
    ap.add_argument("--layers", default="14,15,16,17")
    ap.add_argument("--fusion", default="sum", choices=["sum", "concat"])
    ap.add_argument("--arch", default="gru")
    ap.add_argument("--tag", default="devsel")
    ap.add_argument("--ckpt", default=None, help="explicit .pth path (probe system)")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--prefix", default=None, help="output filename prefix")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    d_in = P.DIM * len(layers) if args.fusion == "concat" else P.DIM
    systems = ["difffalign", "probe"] if args.system == "both" else [args.system]

    results = {}
    for system in systems:
        if system == "difffalign":
            tag = f"aperdiff_diffalign_L{'-'.join(map(str, layers))}_{args.fusion}"
            ckpt = None
        else:
            ckpt = Path(args.ckpt) if args.ckpt else (
                OUT / f"pairprobe_best_L{'-'.join(map(str, layers))}_{args.fusion}_bce_{args.arch}_{args.tag}.pth")
            tag = f"aperdiff_probe_L{'-'.join(map(str, layers))}_{args.fusion}_{args.arch}_{args.tag}"
        prefix = args.prefix or f"{tag}_admin_"
        print(f"\n=== {system} ({tag}) ===", flush=True)

        docs = []
        for lang in LANGS:
            d = load_full(lang, layers)
            print(f"  {lang}: {len(d)} docs", flush=True)
            docs.extend(d)

        if system == "difffalign":
            predict_difffalign(docs, layers, args.fusion)
        else:
            if not ckpt.exists():
                raise FileNotFoundError(f"checkpoint not found: {ckpt}")
            predict_probe(docs, ckpt, layers, args.fusion)

        paths = write_predictions(docs, prefix)
        print("  wrote " + ", ".join(str(p) for p in paths.values()), flush=True)

        for lang in LANGS:
            for split in args.splits.split(","):
                sp, kd = score_split(docs, lang, split)
                results[(system, split, lang)] = (sp, kd)

    summary = {}
    for (system, split, lang), (sp, kd) in results.items():
        summary.setdefault(system, {}).setdefault(split, {})[lang] = {
            "spearman": round(sp, 4), "kendall": round(kd, 4)}
    summary["_meta"] = {"layers": layers, "fusion": args.fusion, "arch": args.arch,
                        "tag": args.tag, "splits": args.splits.split(","),
                        "harness": "nlpstats global correlate (matches evaluate_predictions_admin)"}
    JUDGING.mkdir(parents=True, exist_ok=True)
    summary_path = JUDGING / f"summary_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {summary_path}", flush=True)
    print("\n=== judging-aware metrics (global over tokens, gold -1 dropped) ===")
    for split in args.splits.split(","):
        print(f"\n[{split}]  system              " + "  ".join(f"{l:>7}" for l in LANGS) + "   macro")
        for system in systems:
            rows = [results.get((system, split, l)) for l in LANGS]
            if any(r is None for r in rows):
                continue
            sp = [r[0] for r in rows]
            print(f"  {system:<20} " + "  ".join(f"{v*100:7.2f}" for v in sp)
                  + f"   {np.mean(sp)*100:6.2f}")
            kd = [r[1] for r in rows]
            print(f"  {system+' (Kendall)':<20} " + "  ".join(f"{v*100:7.2f}" for v in kd)
                  + f"   {np.mean(kd)*100:6.2f}")


if __name__ == "__main__":
    main()
