#!/usr/bin/env python3
"""Capture frozen-Apertus features for the iSTS-RSD benchmark (ZurichNLP/SwissGov-RSD).

Writes layerwise/ists_{ROLE}_{lang}_L{14..17}.pt in the same dict format as the SwissGov captures

  ists.py TEST               # eval: ists_de/fr/it from the shipped gold.jsonl (100 each)
  ists.py TRAIN sents        # train: sentence-level  train_{lang}_sents.jsonl     (no augmentation)
  ists.py TRAIN projected    # train: augmented      train_{lang}_projected.jsonl (paper setting, 10k/lang)
"""
import ctypes, os, sys, re, json, time, bisect, urllib.request, numpy as np, torch
from ctypes import POINTER, c_float, c_int, c_size_t, c_uint, c_uint64, c_void_p
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

REPO = Path(__file__).resolve()
while not (REPO / "core").exists():
    REPO = REPO.parent
GGUF = REPO / "core/assets/apertus-v1.5-8b-text-Q5_K_M.gguf"
DYLIB = REPO / "outputs/lib/libelthur.dylib"
OUT = REPO / "python/layerwise"
DATA = REPO / "python/dataset/ists"
GOLD_URL = "https://raw.githubusercontent.com/ZurichNLP/SwissGov-RSD/master/data/evaluation/gold_labels/full/gold.jsonl"
TRAIN_URL = "https://raw.githubusercontent.com/ZurichNLP/SwissGov-RSD/master/data/finetuning/encoder_inputs"
LAYERS = [14, 15, 16, 17]   # NOTE: same layer band as the SwissGov probe / DiffAlign
DIM = 4096
STOP = max(LAYERS)          # NOTE: early-exit the prefill at L17 (see core/genny.zig capture_stop)
WS = re.compile(r"\S+")
_CB = ctypes.CFUNCTYPE(None, c_uint, c_int, c_size_t, c_size_t, POINTER(c_float))


def make_engine():
    """Load libelthur + the model and arm the capture early-exit. (ctypes, as in ablation.py)"""
    os.environ["ELTHUR_MODEL"] = str(GGUF)
    lib = ctypes.CDLL(str(DYLIB))
    lib.model_create_ex.restype = c_void_p
    lib.model_create_ex.argtypes = [ctypes.c_char_p, c_int, ctypes.c_double, ctypes.c_double, c_int]
    lib.model_capture.restype = c_int
    lib.model_capture.argtypes = [c_void_p, POINTER(c_int), c_size_t, c_uint64, c_void_p]
    lib.model_capture_stop.argtypes = [c_void_p, c_int]
    lib.model_free.argtypes = [c_void_p]
    h = lib.model_create_ex(b"apertus", 4, 0.0, 0.9, 40)
    assert h, "model_create_ex failed"
    lib.model_capture_stop(h, STOP)
    return lib, h


def _fetch(path, url):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path)
    return [json.loads(l) for l in open(path) if l.strip()]


def gold_rows():
    return _fetch(DATA / "gold.jsonl", GOLD_URL)


def train_rows(lang, src):
    return _fetch(DATA / f"train_{lang}_{src}.jsonl", f"{TRAIN_URL}/train_{lang}_{src}.jsonl")


def encode(tok, text):
    e = tok(text, add_special_tokens=True, return_offsets_mapping=True)
    ids = list(e["input_ids"]); offs = list(e["offset_mapping"])
    sp = [(m.start(), m.end()) for m in WS.finditer(text)]
    st = [s for s, _ in sp]; s2w = []
    for a, b in offs:
        if b <= a:
            s2w.append(-1); continue
        c = b - 1; i = bisect.bisect_right(st, c) - 1
        s2w.append(i if (i >= 0 and sp[i][0] <= c < sp[i][1]) else -1)
    return ids, s2w, len(sp)


def capture(lib, h, ids):
    n = len(ids)
    out = {l: np.zeros((n, DIM), np.float32) for l in LAYERS}

    def cb(s, l, t, d, p):
        s, l, t = int(s), int(l), int(t)
        if s == 10 and l in out and 0 <= t < n:   # NOTE: site 10 = block-output residual (see ablation.py)
            ctypes.memmove(out[l].ctypes.data + t * DIM * 4, p, int(d) * 4)

    fn = _CB(cb); ci = (c_int * n)(*ids)
    assert lib.model_capture(h, ci, n, 1 << 10, ctypes.cast(fn, c_void_p)) >= 0
    return out


def pool(vec, s2w, nw):
    acc = np.zeros((nw, DIM), np.float32); c = np.zeros((nw, 1), np.float32)
    for t, w in enumerate(s2w):
        if w >= 0:
            acc[w] += vec[t]; c[w, 0] += 1
    c[c == 0] = 1
    return acc / c


def merge_chunks(out_prefix, n):
    if n == 1:
        for l in LAYERS:
            (OUT / f"{out_prefix}_L{l}_part0.pt").rename(OUT / f"{out_prefix}_L{l}.pt")
        return
    for l in LAYERS:
        items = []
        for k in range(n):
            p = OUT / f"{out_prefix}_L{l}_part{k}.pt"
            items.extend(torch.load(p, map_location="cpu", weights_only=False))
            p.unlink()
        tmp = OUT / f"{out_prefix}_L{l}.pt.tmp"
        torch.save(items, tmp)
        tmp.rename(OUT / f"{out_prefix}_L{l}.pt")
        del items


def resume_chunks(out_prefix):
    k = 0
    while all((OUT / f"{out_prefix}_L{l}_part{k}.pt").exists() for l in LAYERS):
        k += 1
    for p in OUT.glob(f"{out_prefix}_L*_part*.pt"):
        if int(p.stem.rsplit("part", 1)[1]) >= k:
            p.unlink()
    return k


def capture_set(rows, out_prefix, lib, h, tok, chunk=800, merge=True):
    t0 = time.time(); skipped = 0
    n_chunks = resume_chunks(out_prefix)          # NOTE: resume after a killed run
    start = n_chunks * chunk
    if start:
        print(f"  resume {out_prefix}: {n_chunks} chunk(s) on disk -> skipping first {start} rows", flush=True)
    per_layer = [[] for _ in LAYERS]

    def flush():
        nonlocal n_chunks
        for j, l in enumerate(LAYERS):
            torch.save(per_layer[j], OUT / f"{out_prefix}_L{l}_part{n_chunks}.pt")
            per_layer[j] = []
        n_chunks += 1

    for r in tqdm(rows[start:], desc=out_prefix, ncols=90):
        ia, sa, na = encode(tok, r["text_a"])
        ib, sb, nb = encode(tok, r["text_b"])
        ya = np.asarray(r["labels_a"], np.float32); yb = np.asarray(r["labels_b"], np.float32)
        if len(ya) != na:
            skipped += 1; continue
        if len(yb) != nb:
            # FIXME: ists_it gold ships a mis-sized labels_b (all -1) —  Italian side is unlabelled pad instead of dropping the pair.
            if np.all(yb < 0):
                yb = np.full(nb, -1.0, np.float32)
            else:
                skipped += 1; continue
        fa = capture(lib, h, ib + ia)   # A sees B
        fb = capture(lib, h, ia + ib)   # B sees A
        for j, l in enumerate(LAYERS):
            A = pool(fa[l][len(ib):], sa, na)
            B = pool(fb[l][len(ia):], sb, nb)
            per_layer[j].append({"feats_a": torch.from_numpy(A).half(),
                                 "feats_b": torch.from_numpy(B).half(),
                                 "labels_a": ya, "labels_b": yb})
        if len(per_layer[0]) >= chunk:
            flush()
    if len(per_layer[0]) > 0:
        flush()
    if merge:
        merge_chunks(out_prefix, n_chunks)
        n = sum(1 for _ in torch.load(OUT / f"{out_prefix}_L{LAYERS[0]}.pt", map_location="cpu", weights_only=False))
        print(f"  saved {out_prefix}_L{LAYERS[0]}..{LAYERS[-1]}.pt  "
              f"({n} pairs, {skipped} skipped, {time.time()-t0:.0f}s)", flush=True)
    return n_chunks


def main():
    role = sys.argv[1] if len(sys.argv) > 1 else "TEST"
    src = sys.argv[2] if len(sys.argv) > 2 else "sents"
    lib, h = make_engine()
    tok = AutoTokenizer.from_pretrained("swiss-ai/Apertus-v1.5-8b", local_files_only=True)
    print(f"model ready; role={role}; src={src}", flush=True)

    if role == "TEST":
        rows = gold_rows()
        print(f"iSTS gold rows: {len(rows)}", flush=True)
        for subset in os.environ.get("SUBSETS", "ists_de,ists_fr,ists_it").split(","):
            sub = [r for r in rows if r.get("subset") == subset]
            if not sub:
                print(f"  !! no rows for subset {subset}", flush=True); continue
            capture_set(sub, f"ists_{role}_{subset.split('_', 1)[1]}", lib, h, tok)
    else:
        # NOTE: keep the two training regimes in separate namespaces sents = _TRAIN_ and projected = _AUG_
        tag = os.environ.get("OUT_TAG") or ("TRAIN" if src == "sents" else "AUG")
        limit = int(os.environ.get("LIMIT", "0"))
        for lang in os.environ.get("LANGS", "de,fr,it").split(","):
            prefix = f"ists_{tag}_{lang}"
            # FIXME: only skip when ALL layers are present (a crash mid-merge must not look 'done').
            done = all((OUT / f"{prefix}_L{l}.pt").exists() for l in LAYERS)
            if done and os.environ.get("FORCE") != "1":
                print(f"skip {lang} (all layers exist); set FORCE=1 to recapture", flush=True)
                continue
            rows = train_rows(lang, src)
            if limit:
                rows = rows[:limit]
            print(f"train_{lang}_{src}: {len(rows)} rows -> {prefix}", flush=True)
            n_chunks = capture_set(rows, prefix, lib, h, tok, merge=False)
            # XXX: free the ~6 GB model BEFORE merging 
            lib.model_free(h)
            print(f"  merging {n_chunks} chunks -> {prefix} (model freed) ...", flush=True)
            merge_chunks(prefix, n_chunks)
            print(f"  saved {prefix}_L{LAYERS[0]}..{LAYERS[-1]}.pt", flush=True)
            lib, h = make_engine()
            print("  model reloaded", flush=True)

    lib.model_free(h)


if __name__ == "__main__":
    main()
