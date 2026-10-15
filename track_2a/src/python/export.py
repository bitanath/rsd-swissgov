#!/usr/bin/env python3
"""
Two outputs, both deterministic and re-runnable -> Final outcome to bake into the exe:

  core/assets/gru.bin        fp16 weight blob for the pair probe -> @embedFile
  app/src/data/swissgov.json + .jsonl  SwissGov-RSD pairs in full-set order for GT/CLI.

Weight blob layout (little-endian):
    magic   [4] "GRU1"
    version u16 = 1
    dtype   u16 = 1 (fp16)
    d_in    u32
    d_model u32
    hidden  u32
    rnn_layers u32
    n_tensors  u32
    per tensor: name_len u16, name bytes, ndim u8, dims[ndim] u32 then, aligned to 8 bytes, the fp16 blobs in table order.
"""
import argparse
import json
import struct
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO / "python/layerwise/pairprobe_best_L14-15-16-17_sum_bce_gru_devsel.pth"
DEFAULT_GRU = REPO / "core/assets/gru.bin"
DEFAULT_DATA = REPO / "app/src/data/swissgov.json"

LANGS = ("de", "fr", "it")
SPLITS = (("holdout", "test"), ("val", "dev")) #TODO: on-disk dataset dir -> benchmark split name shown in the TUI


def export_gru(ckpt_path: Path, out_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    d_in = int(ckpt["d_in"])
    d_model = int(sd["proj.0.bias"].shape[0])
    hidden = int(sd["rnn.bias_hh_l0"].shape[0]) // 3
    rnn_layers = 1 + max(
        int(k.split("_l")[1][0]) for k in sd if k.startswith("rnn.weight_ih_l")
    )

    names = list(sd.keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<4sHHIIIII", b"GRU1", 1, 1, d_in, d_model, hidden, rnn_layers, len(names)))
        for name in names:
            t = sd[name]
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<B", t.dim()))
            for d in t.shape:
                f.write(struct.pack("<I", int(d)))
        pad = (-f.tell()) % 8
        f.write(b"\x00" * pad)
        total = 0
        for name in names:
            t = sd[name].to(torch.float16).contiguous()
            f.write(t.numpy().tobytes())
            total += t.numel()
    print(f"wrote {out_path}  tensors={len(names)} params={total} "
          f"({total * 2 / 1e6:.1f} MB fp16) d_in={d_in} d_model={d_model} "
          f"hidden={hidden} rnn_layers={rnn_layers}")


def export_dataset(data_dir: Path, out_path: Path) -> None:
    """Full-set order: per language merge test+dev and sort by numeric id, matching the
    benchmark's ``gold_labels/full/`` order so the authors' evaluator can zip-and-filter
    a full-set prediction file down to ``--split test``."""
    records = []
    for lang in LANGS:
        merged: dict[str, tuple[str, dict]] = {}
        for disk_split, split in SPLITS:
            path = data_dir / disk_split / f"{lang}.jsonl"
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                merged[r["id"]] = (split, r)
        for rid in sorted(merged, key=lambda i: int(str(i).rsplit("_", 1)[1])):
            split, r = merged[rid]
            records.append({
                "id": r["id"],
                "split": split,
                "lang": lang,
                "text_a": r["text_a"],
                "text_b": r["text_b"],
                "labels_a": r["labels_a"],
                "labels_b": r["labels_b"],
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, separators=(",", ":")))
    jsonl_path = out_path.with_suffix(".jsonl")
    jsonl_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    per = {}
    for r in records:
        per[(r["split"], r["lang"])] = per.get((r["split"], r["lang"]), 0) + 1
    print(f"wrote {out_path} + {jsonl_path.name}  docs={len(records)}  {per}  "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export TUI bake-in artifacts")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--gru-out", type=Path, default=DEFAULT_GRU)
    ap.add_argument("--data-out", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--data-dir", type=Path, default=REPO / "python/dataset/swissgov")
    ap.add_argument("--skip-gru", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()

    if not args.skip_gru:
        export_gru(args.ckpt, args.gru_out)
    if not args.skip_data:
        export_dataset(args.data_dir, args.data_out)


if __name__ == "__main__":
    main()
