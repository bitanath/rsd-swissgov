#!/usr/bin/env python3
"""Reproduce the SwissGov-RSD DiffAlign baseline (we also use diffalign.py as a copy-paste reimplementation)
This is just a formality

Usage:
    SWISSGOV_RSD=/path/to/SwissGov-RSD python reproduce_baseline.py \
        --layers 14,15,16,17 --fusion sum
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "python" / "layerwise"
LANGS = ("de", "fr", "it")
PINNED = "1807a42100e742ed03d337c54c4b9ea86995f565"
MODEL_LABEL = "swiss-ai_Apertus-v1.5-8B"


def rsd_repo(explicit):
    if explicit:
        return Path(explicit)
    if os.environ.get("SWISSGOV_RSD"):
        return Path(os.environ["SWISSGOV_RSD"])
    cache = REPO / "python" / ".cache" / "SwissGov-RSD"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/ZurichNLP/SwissGov-RSD.git", str(cache)],
                       check=True)
    subprocess.run(["git", "-C", str(cache), "checkout", PINNED], check=True)
    return cache


def load_lang(lang, layers, fusion):
    """Return {id: (text_a, text_b, A, B)} with word-level fused Apertus features."""
    role_docs = {}
    for role in ("train", "val"):            # dev(168) + test(56) = full 224
        per = [torch.load(OUT / f"{role}_{lang}_L{l}.pt", map_location="cpu", weights_only=False)
               for l in layers]
        for i in range(len(per[0])):
            e = per[0][i]
            fa = [per[j][i]["feats_a"].float() for j in range(len(layers))]
            fb = [per[j][i]["feats_b"].float() for j in range(len(layers))]
            if fusion == "sum":
                A = sum(fa) / len(layers); B = sum(fb) / len(layers)
            else:
                A = torch.cat(fa, -1); B = torch.cat(fb, -1)
            role_docs[e["id"]] = (e["tokens_a"], e["tokens_b"], A, B)
        del per
    return role_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsd-repo", default=None)
    ap.add_argument("--layers", default="14,15,16,17")
    ap.add_argument("--fusion", default="sum", choices=["sum", "concat"])
    ap.add_argument("--split", default="full", choices=["full", "dev", "test"])
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    RSD = rsd_repo(args.rsd_repo)
    sys.path.insert(0, str(RSD))
    from rsd.recognizers.diff_align import DiffAlign        # noqa: E402
    from evaluation.utils import load_gold_data             # noqa: E402

    class CachedApertusDiffAlign(DiffAlign):
        """The authors' DiffAlign with a frozen-Apertus word-level encoder."""

        def __init__(self, lookup):
            # deliberately bypass FeatureExtractionRecognizer.__init__ (no model load)
            self.lookup = lookup
            self.layer = -1
            self.batch_size = 4
            self.model_name_or_path = MODEL_LABEL
            self.pipeline = None
            self.is_sentence_transformer = False
            self.is_mmbert = False

        def encode_batch(self, sentences, **kwargs):
            return [self.lookup[s] for s in sentences]   # (n_words, dim), no specials

        def _get_subwords_by_word(self, sentence):
            return [[i] for i in range(len(sentence.split()))]   # word-level == 1 subword each

    # Always predict the FULL set (224/lang, full-set order) and let the authors'
    # harness filter to the split.
    out_dir = RSD / "data" / "evaluation" / "encoder_predictions" / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"DiffAlign(model={MODEL_LABEL}, layer=-1"
    for lang in LANGS:
        lookup = load_lang(lang, layers, args.fusion)
        feats = {}                              # text -> word-level Apertus vector
        for _id, (ta, tb, A, B) in lookup.items():
            feats[" ".join(ta)] = A
            feats[" ".join(tb)] = B
        rec = CachedApertusDiffAlign(feats)

        gold_file = RSD / "data" / "evaluation" / "gold_labels" / "full" / f"gold_admin_{lang}.jsonl"
        gold = load_gold_data(gold_file)
        gold_ids = [json.loads(l)["id"] for l in open(gold_file) if l.strip()]
        out = out_dir / f"{tag.replace('/', '_')}_admin_{lang}.jsonl.jsonl"
        with open(out, "w") as f:
            for gid, g in zip(gold_ids, gold):
                p = rec.predict(" ".join(g.tokens_a), " ".join(g.tokens_b))
                assert p.tokens_a == g.tokens_a and p.tokens_b == g.tokens_b
                f.write(json.dumps({
                    "id": gid,
                    "text_a": " ".join(p.tokens_a),
                    "text_b": " ".join(p.tokens_b),
                    "labels_a": [round(float(x), 5) for x in p.labels_a],
                    "labels_b": [round(float(x), 5) for x in p.labels_b],
                }) + "\n")
        print(f"wrote {out}  ({len(gold)} samples)")

    prefix = str(out_dir / f"{tag.replace('/', '_')}_admin_")
    env = {**os.environ, "PYTHONPATH": str(RSD)}
    for split in ["test", "dev", "full"]:
        print(f"\n########## evaluate_predictions_admin --split {split} ##########")
        subprocess.run([sys.executable, str(RSD / "scripts" / "evaluate_predictions_admin.py"),
                        prefix, "--split", split], cwd=str(RSD), env=env, check=False)


if __name__ == "__main__":
    main()
