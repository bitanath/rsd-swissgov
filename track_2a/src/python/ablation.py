#!/usr/bin/env python3
import argparse
import ctypes
import glob
import json
import os
from ctypes import POINTER, c_double, c_float, c_int, c_size_t, c_uint, c_uint64, c_void_p
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
GGUF = REPO / "core/assets/apertus-v1.5-8b-text-Q5_K_M.gguf"
DYLIB = REPO / "outputs/lib/libelthur.dylib"
TOKENIZER_REPO = "swiss-ai/Apertus-v1.5-8B"

SITE_RES_FFN = 10          # capture site: block output residual stream
DIM = 4096                 # Apertus d_model
HIGH = 0.5                 # "different meaning" threshold
LOW = 0.05                 # "same meaning" threshold


def heuristic_name(layer: int) -> str:
    if layer <= 3:
        return "Surface & orthography"
    if layer <= 9:
        return "Local syntax / morphology"
    if layer <= 15:
        return "Semantic build-up"
    if layer == 16:
        return "Mid-depth semantic core"
    if layer <= 21:
        return "Cross-lingual semantics"
    if layer <= 29:
        return "Task & discourse"
    return "Output formation"


_CB = ctypes.CFUNCTYPE(None, c_uint, c_int, c_size_t, c_size_t, POINTER(c_float))


def make_engine():
    os.environ["ELTHUR_MODEL"] = str(GGUF)
    lib = ctypes.CDLL(str(DYLIB))
    lib.model_create_ex.restype = c_void_p
    lib.model_create_ex.argtypes = [ctypes.c_char_p, c_int, c_double, c_double, c_int]
    lib.model_capture.restype = c_int
    lib.model_capture.argtypes = [c_void_p, POINTER(c_int), c_size_t, c_uint64, c_void_p]
    lib.model_free.argtypes = [c_void_p]
    h = lib.model_create_ex(b"", 4, 0.0, 0.9, 40)
    assert h, "model_create_ex failed"
    return lib, h


def capture_all_layers(lib, h, ids, n_layers):
    """Return (n_layers, n_tokens, DIM) residual-stream features for `ids`."""
    n = len(ids)
    out = np.zeros((n_layers, n, DIM), np.float32)
    base = out.ctypes.data
    stride_tok = DIM * 4

    def cb(site, layer, tok, dim, ptr):
        s, l, t = int(site), int(layer), int(tok)
        if s == SITE_RES_FFN and 0 <= l < n_layers and 0 <= t < n:
            ctypes.memmove(base + (l * n + t) * stride_tok, ptr, int(dim) * 4)

    fn = _CB(cb)
    ci = (c_int * n)(*ids)
    rc = lib.model_capture(h, ci, n, 1 << SITE_RES_FFN, ctypes.cast(fn, c_void_p))
    assert rc >= 0, "model_capture failed"
    return out


_WS = None  #NOTE set in main


def encode(tok, text):
    import re
    global _WS
    if _WS is None:
        _WS = re.compile(r"\S+")
    e = tok(text, add_special_tokens=True, return_offsets_mapping=True)
    ids = list(e["input_ids"])
    offs = list(e["offset_mapping"])
    spans = [(m.start(), m.end()) for m in _WS.finditer(text)]
    starts = [s for s, _ in spans]
    s2w = []
    import bisect
    for a, b in offs:
        if b <= a:
            s2w.append(-1)
            continue
        c = b - 1
        i = bisect.bisect_right(starts, c) - 1
        s2w.append(i if (i >= 0 and spans[i][0] <= c < spans[i][1]) else -1)
    return ids, s2w, len(spans)


def pool_matrix(s2w, n_tokens, n_words):
    """Dense (n_words, n_tokens) averaging matrix over valid tokens."""
    P = np.zeros((n_words, n_tokens), np.float32)
    for t, w in enumerate(s2w):
        if w >= 0:
            P[w, t] = 1.0
    counts = P.sum(1, keepdims=True)
    counts[counts == 0] = 1.0
    return P / counts


def pair_features(lib, h, tok, row, n_layers):
    """Return per-layer word features + gold labels for both sides of a pair."""
    ia, sa, na = encode(tok, row["text_a"])
    ib, sb, nb = encode(tok, row["text_b"])

    # HACK A sees B: sequence [B][A], keep A's positions
    full = capture_all_layers(lib, h, ib + ia, n_layers)
    out_ab = np.ascontiguousarray(full[:, len(ib):, :])
    del full
    Xa = _pool_layers(out_ab, sa, na, n_layers)
    del out_ab

    # HACK B sees A: sequence [A][B], keep B's positions
    full = capture_all_layers(lib, h, ia + ib, n_layers)
    out_ba = np.ascontiguousarray(full[:, len(ia):, :])
    del full
    Xb = _pool_layers(out_ba, sb, nb, n_layers)
    del out_ba

    ya = np.asarray(row["labels_a"][:na], np.float32)
    yb = np.asarray(row["labels_b"][:nb], np.float32)
    return Xa, ya, Xb, yb


def _pool_layers(out, s2w, nw, n_layers):
    n = out.shape[1]
    P = pool_matrix(s2w, n, nw)
    X = np.empty((n_layers, nw, DIM), np.float32) # out: (L, n, D) contiguous -> (L, nw, D) word means
    for l in range(n_layers):
        X[l] = P @ out[l]
    return X


def auc_score(scores, labels):
    pos = labels == 1
    neg = labels == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return (r[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def main():
    ap = argparse.ArgumentParser(description="Training-free layer ablation for Apertus-RSD")
    ap.add_argument("--split", default="full", choices=["full", "val", "holdout"])
    ap.add_argument("--limit", type=int, default=20, help="pairs per language")
    ap.add_argument("--calib-frac", type=float, default=0.3, help="fraction of pairs used to estimate d")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated indices")
    ap.add_argument("--outdir", default=str(REPO / "python"))
    ap.add_argument("--out", default="ablation.json",
                    help="output filename under --outdir (use a new name to avoid overwriting)")
    args = ap.parse_args()

    n_layers = 32
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    data_dir = REPO / f"python/dataset/swissgov/{args.split}"
    rows = []
    for lang in ("de", "fr", "it"):
        with open(data_dir / f"{lang}.jsonl") as f:
            recs = [json.loads(l) for l in f if l.strip()][: args.limit]
        for r in recs:
            r["_lang"] = lang
        rows.extend(recs)
    
    per_lang = {}
    for r in rows:
        per_lang.setdefault(r["_lang"], []).append(r)
    calib, evalset = [], []
    for lang, recs in per_lang.items():
        k = max(1, int(round(len(recs) * args.calib_frac)))
        calib.extend(recs[:k])
        evalset.extend(recs[k:])

    print(f"device: cpu | split: {args.split} | pairs calib/eval: {len(calib)}/{len(evalset)} "
          f"| layers: {len(layers)}", flush=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO, local_files_only=True)
    lib, h = make_engine()

    #  TODO AI suggested calibration accumulators:
    #   OLS  (continuous-label linear direction)  -> d_ols = Cov(x,y)
    #   DoM  (difference of class means)          -> d_dom = mu_hi - mu_lo
    Sx = np.zeros((n_layers, DIM), np.float64)
    Sxy = np.zeros((n_layers, DIM), np.float64)
    sum_hi = np.zeros((n_layers, DIM), np.float64)
    sum_lo = np.zeros((n_layers, DIM), np.float64)
    cnt_hi = np.zeros(n_layers, np.int64)
    cnt_lo = np.zeros(n_layers, np.int64)
    sy = 0.0
    ntot = 0

    for row in tqdm(calib, desc="calibrate", unit="pair", ncols=90, file=__import__("sys").stdout):
        Xa, ya, Xb, yb = pair_features(lib, h, tok, row, n_layers)
        for X, y in ((Xa, ya), (Xb, yb)):
            m = y >= 0
            X = X[:, m, :]
            y = y[m]
            if not len(y):
                continue
            Sx += X.sum(1)
            Sxy += np.einsum("lwd,w->ld", X, y)
            sy += float(y.sum())
            ntot += len(y)
            hi = y >= HIGH
            lo = y <= LOW
            if hi.any():
                sum_hi += X[:, hi, :].sum(1)
                cnt_hi += hi.sum()
            if lo.any():
                sum_lo += X[:, lo, :].sum(1)
                cnt_lo += lo.sum()

    d_ols = np.zeros((n_layers, DIM), np.float64)
    d_dom = np.zeros((n_layers, DIM), np.float64)
    for l in range(n_layers):
        if ntot > 0:
            d_ols[l] = Sxy[l] - Sx[l] * (sy / ntot)
            nn = np.linalg.norm(d_ols[l])
            if nn > 0:
                d_ols[l] /= nn
        if cnt_hi[l] > 0 and cnt_lo[l] > 0:
            d_dom[l] = sum_hi[l] / cnt_hi[l] - sum_lo[l] / cnt_lo[l]
            nn = np.linalg.norm(d_dom[l])
            if nn > 0:
                d_dom[l] /= nn

    scores = {l: [] for l in range(n_layers)}     
    scores_dom = {l: [] for l in range(n_layers)} 
    labels = {l: [] for l in range(n_layers)}
    scores_lang = {(l, g): [] for l in range(n_layers) for g in ("de", "fr", "it")}
    labels_lang = {(l, g): [] for l in range(n_layers) for g in ("de", "fr", "it")}

    for row in tqdm(evalset, desc="evaluate ", unit="pair", ncols=90, file=__import__("sys").stdout):
        Xa, ya, Xb, yb = pair_features(lib, h, tok, row, n_layers)
        for X, y in ((Xa, ya), (Xb, yb)):
            m = y >= 0
            if not m.any():
                continue
            X = X[:, m, :]
            y = y[m]
            g = row["_lang"]
            for l in layers:
                s = X[l] @ d_ols[l]
                sd = X[l] @ d_dom[l]
                scores[l].append(s)
                scores_dom[l].append(sd)
                labels[l].append(y)
                scores_lang[(l, g)].append(s)
                labels_lang[(l, g)].append(y)

    lib.model_free(h)

    results = []
    for l in layers:
        s = np.concatenate(scores[l]).astype(np.float64)
        sd = np.concatenate(scores_dom[l]).astype(np.float64)
        y = np.concatenate(labels[l]).astype(np.float64)
        rho = spearmanr(s, y).statistic if len(s) > 2 else float("nan")
        r = pearsonr(s, y).statistic if len(s) > 2 else float("nan")
        rho_dom = spearmanr(sd, y).statistic if len(sd) > 2 else float("nan")
        bin_y = np.where(y >= HIGH, 1, np.where(y <= LOW, 0, -1))
        aucv = auc_score(sd[bin_y >= 0], bin_y[bin_y >= 0])
        per_lang = {}
        for g in ("de", "fr", "it"):
            sg = np.concatenate(scores_lang[(l, g)]).astype(np.float64)
            yg = np.concatenate(labels_lang[(l, g)]).astype(np.float64)
            per_lang[g] = spearmanr(sg, yg).statistic if len(sg) > 2 else float("nan")
        hi, lo = sd[bin_y == 1], sd[bin_y == 0]
        sep = (hi.mean() - lo.mean()) / (sd.std() + 1e-9) if len(hi) and len(lo) else float("nan")
        results.append(dict(layer=l, name=heuristic_name(l), spearman=rho, pearson=r, spearman_dom=rho_dom,
                            auc=aucv, sep=sep, de=per_lang["de"], fr=per_lang["fr"], it=per_lang["it"],
                            n=int(len(s))))

    header = ("| Layer | Heuristic name | Spearman ρ (OLS) | Pearson r (OLS) | ρ (DoM) | AUC (DoM) | Sep | ρ de | ρ fr | ρ it | n |\n"
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines = [header]
    best = max(results, key=lambda x: (x["spearman"] if x["spearman"] == x["spearman"] else -9))
    best_auc = max(results, key=lambda x: (x["auc"] if x["auc"] == x["auc"] else -9))
    for x in results:
        star = " ★" if x["layer"] == best["layer"] else ""
        star_auc = " ◆" if x["layer"] == best_auc["layer"] else ""
        lines.append(f"| {x['layer']}{star}{star_auc} | {x['name']} | {x['spearman']:.4f} | {x['pearson']:.4f} | "
                     f"{x['spearman_dom']:.4f} | {x['auc']:.4f} | {x['sep']:.3f} | "
                     f"{x['de']:.4f} | {x['fr']:.4f} | {x['it']:.4f} | {x['n']} |")
    table = "\n".join(lines)

    print("\n" + table + "\n")
    print(f"best layer by Spearman: {best['layer']} ({best['name']})  rho={best['spearman']:.4f}", flush=True)

    outdir = Path(args.outdir)
    (outdir / args.out).write_text(json.dumps(
        dict(layers=results, best=best["layer"], split=args.split, limit=args.limit,
             calib=len(calib), eval=len(evalset)), indent=2))
    print(f"wrote {outdir/args.out}  (table also printed above)")


if __name__ == "__main__":
    main()
