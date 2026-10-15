# aperdiff — Recognition of Semantic Differences on SwissGov-RSD

Cross-lingual **Recognition of Semantic Differences (RSD)** over Swiss government
documents, built on a **frozen** [`swiss-ai/Apertus-v1.5-8B`](https://huggingface.co/swiss-ai/Apertus-v1.5-8B)
residual stream and a light trained probe. It ships as a self-contained TUI +
headless CLI whose only dependency is a container runtime.

> **HackApertus — Track 2A (Academia):** *Detecting Cross-Lingual Semantic
> Differences in Swiss Government Websites* (UZH).

## Local · CPU-only · sovereign

- **Runs entirely locally.** No hosted endpoint, no API keys, no network access at
  inference — the Apertus weights travel inside the image/binary.
- **CPU-only.** No GPU, no CUDA, no Metal. ~5–6 GB RAM, SIMD-optimized for ARM
  (Apple Silicon / ARM Linux); x86-64 works but is slower.
- **Sovereign by construction.** Fits on-premise, air-gapped, and Swiss-cloud
  deployments: nothing leaves the machine.
- **Frozen model.** Apertus is never fine-tuned; only a small pair-probe head is
  trained on `dev`, and the scored `test` split is never used for selection.

## Interactive predictions browser

The TUI lets you *touch and feel* the predictions: browse the 224 document pairs
per language, jump across the scored `test` split, and toggle which head drives
the highlights.

**Supervised pair-probe head** (`ctrl+r`) — per-token difference scores from the
trained GRU probe, with a grounded English note:

![Interactive TUI — supervised pair-probe head](track_2a/docs/interactive_tui.png)

**Combined head** (`ctrl+d`) — the union of the supervised and unsupervised
(DiffAlign) signals, plus the command palette (`/supervisedalign`, `/diffalign`,
`/combined`, `/highlights`, `/groundtruth`, `/theme`, `/help`, `/quit`):

![Interactive TUI — combined heads and command palette](track_2a/docs/interactive_tui_combined_heads.png)

## Quickstart

Judges run a single command from the repository root:

```bash
make run
```

`make run` pulls the published multi-arch image and prints the **bundled
results** table. Everything else is selected with variables — the CLI arguments
are exactly the container's arguments:

| Invocation | Container argv |
| --- | --- |
| `make run` | `results` |
| `make run MODE=predict HEAD=unsupervised LANG=it TEST=1` | `predict --head unsupervised --lang it --test --output outputs/predictions` |
| `make run MODE=predict INPUT=custom.jsonl OUTPUT=out JSON=1` | `predict --input custom.jsonl --output out --json` |
| `make run MODE=results LANG=de JSON=1` | `results --lang de --json` |
| `make run MODE=tui` | TUI (`-it`, `TERM=xterm-256color`, `--detach-keys=""`) |
| `make run MODE=tui ARGS="--texta '…' --textb '…' --language de --note x"` | TUI direct comparison |
| `make run MODE=smoke SMOKE_FAST=1` | `--smoke --smoke-fast` |
| `make run MODE=speedtest` | `--speedtest` |
| `make run ARGS="<anything>"` | verbatim — any subcommand/flag incl. `--bench`, `results --json`, etc. |

Variables: `IMAGE` (default `ghcr.io/bitanath/rsd-swissgov:latest`), `PLATFORM`,
`MODE`, `HEAD`, `LANG`, `INPUT`, `OUTPUT`, `TEST`, `JSON`, `SMOKE_FAST`, `ARGS`.
`LLM_NAME` / `LLM_BASE_URL` / `LLM_API_KEY` are forwarded to the container (the
model is local, so they are informational).

The `results` / `predict` tables carry the paper's token-level metrics
**per document** — `spearman` and `kendall` (τ-b) against the gold labels, pooled
over both texts with gold `-1` dropped.

### Equivalent raw Docker commands

```bash
# reproduce the bundled full-set predictions (supervised head) -> ./predictions.jsonl
docker run --rm -v "$PWD:/data" ghcr.io/bitanath/rsd-swissgov:latest predict --output predictions

# predict an arbitrary jsonl ({id,text_a,text_b}; gold labels optional)
docker run --rm -v "$PWD:/data" ghcr.io/bitanath/rsd-swissgov:latest predict --input custom.jsonl --output outputs

# unsupervised head (DiffAlign)
docker run --rm -v "$PWD:/data" ghcr.io/bitanath/rsd-swissgov:latest predict --input custom.jsonl --head unsupervised --output outputs

# restrict languages; only the scored test split (56/lang)
docker run --rm -v "$PWD:/data" ghcr.io/bitanath/rsd-swissgov:latest predict --lang de,fr --output outputs
docker run --rm -v "$PWD:/data" ghcr.io/bitanath/rsd-swissgov:latest predict --test --output outputs

# precomputed predictions (the numbers in our technical report)
docker run --rm ghcr.io/bitanath/rsd-swissgov:latest results
docker run --rm ghcr.io/bitanath/rsd-swissgov:latest results --lang it

# self-check and throughput probe
docker run --rm ghcr.io/bitanath/rsd-swissgov:latest --smoke
docker run --rm ghcr.io/bitanath/rsd-swissgov:latest --speedtest

# interactive TUI
docker run --rm -it --detach-keys="" -e TERM=xterm-256color ghcr.io/bitanath/rsd-swissgov:latest
```

## Results

SwissGov-RSD **test** split (56 docs/lang), token-level **Spearman ×100**, gold
`-1` dropped. Full table and the paper comparison are in
[`technical_report.md`](technical_report.md).

| System | EN–DE | EN–FR | EN–IT | macro |
| --- | --: | --: | --: | --: |
| DiffAlign · Apertus L15 (unsupervised) | 32.2 | 18.2 | 36.0 | 28.8 |
| DiffAlign · Apertus L14–17 sum (unsupervised) | 31.5 | 18.4 | 36.6 | 28.8 |
| **Pair probe · Apertus L14–17 → GRU (supervised)** | **44.1** | **27.2** | **47.3** | **39.5** |
| Human annotator agreement (ceiling) | 56.1 | 43.6 | 65.1 | 54.9 |

## Build

The image is built from a **multi-stage** `Dockerfile` at the repository root:
the Bun/OpenTUI app is compiled against the prebuilt engine libraries
(`track_2a/src/libelthur-*.so`), the Apertus GGUF is fetched from Hugging Face,
and the runtime points the engine at it via `ELTHUR_MODEL`.

```bash
make build                 # build for the host architecture -> aperdiff:local
make run IMAGE=aperdiff:local
make push                  # multi-arch (linux/amd64 + linux/arm64) build & push
```

> Building both architectures needs a container runtime able to serve both
> (`colima` with Rosetta on Apple Silicon). The `Dockerfile` is written so that
> every `RUN` executes on the **build** platform and the target stages are
> `COPY`-only — no QEMU emulation is required.

## Hardware & performance

Host = Apple Silicon, Apertus 1.5 8B Q5_K_M: **~16 tok/s decode**, **~63 tok/s
prefill** (in-container ~12.5 / ~48). The first launch is slower while the OS
validates the binary. `--speedtest` reports the live numbers and flags an
underpowered machine.

## Repository layout

| Path | What it is |
| --- | --- |
| `README.md` | this file — build, run, and results |
| `technical_report.md` | the write-up: architecture, data, evaluation, ablation |
| `Makefile` | `make run` and friends |
| `Dockerfile` | multi-stage build (app + engine libs + GGUF) |
| `track_2a/src/app/` | the OpenTUI (Bun/TypeScript) application: TUI + headless CLI |
| `track_2a/src/python/` | training / ablation / judging scripts and checkpoints |
| `track_2a/src/libelthur-*.so`, `libelthur.dylib` | prebuilt CPU engine libraries |
| `track_2a/data/` | the SwissGov-RSD pairs and the bundled predictions |
| `track_2a/docs/` | screenshots and diagrams |
| `track_2a/README.md` | code-level details |

## License

Apache-2.0 (code). The SwissGov-RSD labels are released under CC BY; the
copyright of the underlying `admin.ch` text remains with the Swiss federal
authorities.
