# aperdiff — code details (Track 2A)

This directory holds the source of the submission. The build/run instructions and
the results live in the repository-root [`README.md`](../README.md); the write-up
is in [`technical_report.md`](../technical_report.md).

## Layout

```
src/
├── app/                 Bun + OpenTUI (TypeScript) application
│   ├── index.tsx        entry point: speedtest → smoke → results/predict → TUI
│   └── src/
│       ├── terminalexec.ts   headless CLI: `results` / `predict`
│       ├── smoke.ts          --smoke self-check
│       ├── speedtest.ts      --speedtest throughput probe
│       ├── funcs.ts          bun:ffi bindings to the engine (libelthur)
│       ├── interface/        TUI screens (doc browser, pair view, notes)
│       └── libraries/
│           ├── clit/         headless helpers: analyze, dataset, metrics
│           ├── hunk/         vendored diff/rendering layer (OpenTUI)
│           └── progress/     worker-based per-token progress
├── python/              training / ablation / judging (PyTorch)
│   ├── probationists.py      pair-probe architectures + training
│   ├── ablation.py           training-free 32-layer readout
│   ├── judging.py            SwissGov-RSD harness replication
│   ├── diffalign.py          unsupervised DiffAlign head
│   ├── export.py             bake artifacts: gru.bin + swissgov.json(l)
│   └── layerwise/            checkpoints (.pth) + dataset/ (SwissGov, iSTS)
├── libelthur-arm64.so   prebuilt CPU engine (Linux arm64)
├── libelthur-amd64.so   prebuilt CPU engine (Linux amd64)
└── libelthur.dylib      prebuilt CPU engine (macOS arm64)
```

## How it works

Two heads over one frozen backbone (see `python/ABLATION.md` for the research
narrative):

- **Capture.** Apertus is causal, so each side is read with the other side in
  context using two passes: `[B][A]` keeps the English rows, `[A][B]` keeps the
  other-language rows. Only residual sites **L14–17** are captured, and a
  capture-only early exit after L15 makes this ~2× faster with bit-identical L15
  vectors.
- **DiffAlign (unsupervised).** `s = 1 − max_j cos(h_i, h_j)` — the maximum
  cosine similarity to any token on the other side; training-free.
- **Pair probe (supervised).** `[EN;XX]` token features are concatenated with a
  segment marker and scored by a small bidirectional GRU (projection → LayerNorm
  → GELU → GRU → per-token head), trained with BCE. Only these ~2M parameters are
  learned. Architecture chosen by a `dev` screen (`probationists.py`); the final
  checkpoint is `layerwise/pairprobe_best_L14-15-16-17_sum_bce_gru_devsel.pth`.

## Headless CLI

`app/src/terminalexec.ts` implements the judge-facing surface. In the container:

```
results [--lang de,fr,it|all] [--test] [--json]
predict [--head supervised|unsupervised|combined] [--lang …] \
        [--input file.jsonl] [--output name] [--test] [--json]
--smoke [--smoke-fast]      --speedtest (alias --bench)
```

The `results`/`predict` tables print the paper's **per-document** metrics
(`spearman`, `kendall`), computed in `app/src/libraries/clit/metrics.ts` with the
same semantics as the SwissGov-RSD harness (pooled over both texts, gold `-1`
dropped, predictions aligned to gold length).

## Engine libraries

`libelthur-*.so` / `libelthur.dylib` are the compiled CPU engine (the Zig `core/`
of the upstream `aperdiff` project). They are shipped prebuilt so the app can be
compiled without a Zig toolchain. The Docker build stages the matching library as
`app/libelthur.bin` before `bun build --compile`.

## Python pipeline

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r python/requirements.txt

python python/ablation.py --split val --limit 20 --calib-frac 0.35 --out ablation_devsel.json
python python/judging.py --system probe --tag devsel --arch gru
python python/export.py            # -> core/assets/gru.bin + app/src/data/swissgov.json(l)
```

Feature caches (`python/layerwise/*_L14..17.pt`) are large and are **not** checked
in; regenerate them with the capture scripts before training.
