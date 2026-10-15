# Technical report — aperdiff: Recognition of Semantic Differences on SwissGov-RSD

> Cross-lingual Recognition of Semantic Differences (RSD) on Swiss government
> documents, solved with a **frozen** `Apertus-v1.5-8B` residual stream and a light
> trained probe. It runs **locally on CPU only** (~5–6 GB RAM), needs no hosted
> endpoint, and beats every published DiffAlign encoder on the naturalistic
> SwissGov-RSD benchmark while reaching the top of the paper's fine-tuned-encoder
> band on synthetic iSTS-RSD.

- **Track:** Apertus Readiness — Track 2A (Academia): UZH, *Detecting Cross-Lingual Semantic Differences in Swiss Government Websites*
- **Event:** Online
- **Team:** aperdiff
- **Demo:** `make run` (headless results) and `make run MODE=tui` (interactive predictions browser)

## 1. Summary

Recognizing semantic differences (RSD) asks which tokens of one text differ in
meaning from a related text — here, an English `admin.ch` page and its German,
French, or Italian counterpart. We keep **Apertus-v1.5-8B frozen** and read its
intermediate residual stream: an **unsupervised** DiffAlign head
(cosine max-alignment over per-token features) and a small **supervised**
bidirectional-GRU pair-probe trained on a held-out `dev` split. On the scored
SwissGov-RSD `test` split our frozen-Apertus DiffAlign reaches a macro Spearman of
**28.8** (best published DiffAlign encoder: LaBSE ≈ 18.1), and the pair-probe
reaches **44.1 / 27.2 / 47.3** (macro **39.5**) against a human-agreement ceiling of
**56.1 / 43.6 / 65.1**. Everything runs offline, CPU-only.

## 2. Results

SwissGov-RSD **test** split (56 docs/lang), token-level **Spearman ×100**, gold
`-1` dropped. `(ours)` rows use frozen Apertus; paper rows are reproduced from
Wastl, Vamvas & Sennrich (ACL 2026), Table 3.

| System | trained on | EN–DE | EN–FR | EN–IT | macro |
| --- | --- | --: | --: | --: | --: |
| DiffAlign · Apertus L15 *(ours)* | — | 32.2 | 18.2 | 36.0 | 28.8 |
| DiffAlign · Apertus L14–17 sum *(ours)* | — | 31.5 | 18.4 | 36.6 | 28.8 |
| **Pair probe · Apertus L14–17 → GRU *(ours)*** | SwissGov in-domain | **44.1** | **27.2** | **47.3** | **39.5** |
| DiffAlign · LaBSE *(paper)* | — | 20.6 | 8.2 | 25.6 | 18.1 |
| DiffAlign · XLM-R + SimCSE *(paper)* | — | 19.2 | 10.0 | 23.4 | 17.5 |
| DiffAlign · bge-m3 *(paper)* | — | 14.8 | 9.9 | 13.6 | 12.8 |
| GPT-4o *(paper, few-shot)* | — | 4.2 | 1.2 | 5.5 | 3.6 |
| ModernBERT-large (EN) *(paper, fine-tuned)* | iSTS | 5.7 | 1.3 | 4.0 | 3.7 |
| **Human annotator agreement (ceiling)** | — | **56.1** | **43.6** | **65.1** | **54.9** |

Kendall τ-b (the paper's robustness check) tracks Spearman: macro **23.2** for
DiffAlign·L15, **23.2** for L14–17, and **32.2** for the pair-probe on `test`.

**Reading.** The paper's own finding is a large *generalization gap*: systems that
score ~45–82 on synthetic iSTS-RSD collapse to ~0–26 on naturalistic SwissGov-RSD.
Our in-domain, frozen-Apertus probe closes a large part of that gap and — unlike
the paper's fine-tuned rows, which train on synthetic data and transfer zero-shot —
is trained only on SwissGov `dev`. The `test` split is used exactly once, for
reporting. The CLI exposes these metrics **per document** (`spearman`, `kendall`
columns in `results`/`predict`).

## 3. Architecture

Two heads over one frozen backbone, all CPU:

```
EN text ─┐                                   ┌─ DiffAlign (unsupervised)
         ├─ Apertus-v1.5-8B (frozen) ─ feats ─┤   s = 1 − max_j cos(h_i, h_j)
XX text ─┘   capture L14–17                  └─ Pair-probe (supervised GRU)
                                                 concat[EN;XX] + seg → 1 score/token
```

- **Capture.** Because Apertus is causal, each side is read with the *other* side in
  context using two passes (`[B][A]` keeps English; `[A][B]` keeps the other
  language). Only residual-stream sites **L14–17** are captured; a capture-only
  early exit stops the layer loop after L15 for speed with bit-identical vectors.
- **Unsupervised head (DiffAlign).** For every token, `1 − max cosine similarity`
  to any token on the other side — training-free.
- **Supervised head (pair probe).** The two token sequences are concatenated with a
  segment marker and passed through a small bidirectional GRU (projection →
  LayerNorm → GELU → GRU → per-token head), trained with BCE. Only these ~2M
  parameters are learned; Apertus stays frozen. The final architecture (GRU) was
  chosen by a `dev` screen against CNN / bilinear / MLP / transformer / cross-attn
  alternatives; it is a near-tie with CNN and is retained for its per-language
  balance.
- **Interactive TUI + headless CLI.** A Bun/OpenTUI front-end (`track_2a/src/app`)
  over a prebuilt CPU engine (`libelthur`); the same binary serves `results`,
  `predict`, `--smoke`, `--speedtest`, and the TUI.

## 4. Use of Apertus

- **Model:** `swiss-ai/Apertus-v1.5-8B` (text, Q5_K_M GGUF).
- **How it is used:** *inference only* — the weights are frozen; we read the
  residual stream. No fine-tuning, no LoRA, no adapters.
- **Where it runs:** local weights, embedded in the image/binary. No hosted
  endpoint, no network at inference.
- **Footprint:** ~5–6 GB RAM, CPU-only, SIMD-optimized for ARM.

## 5. Data

- **SwissGov-RSD** (Wastl et al., 2026): 224 multi-parallel EN–DE / EN–FR / EN–IT
  document pairs from `admin.ch`, with human token-level difference labels. We use
  the released `dev` (168/lang) for training/selection and the **`test` split
  (56/lang)** for scoring only. Bundled compactly in `track_2a/data/swissgov.jsonl`
  (and `track_2a/src/python/dataset/swissgov/`).
- **iSTS-RSD** (Vamvas & Sennrich, 2023): synthetic, sentence/document-level;
  used only for the secondary, zero-shot transfer numbers and the `projected`
  augmentation used to train the iSTS probe. Its label projection was produced by
  the benchmark authors with GPT-4o-mini; that model is **not** part of our system.
- **Licensing.** SwissGov-RSD labels are CC BY; the copyright of the underlying
  `admin.ch` text remains with the Swiss federal authorities.

## 6. Evaluation

**Protocol.** Token-level regression scored by **global Spearman** over all tokens
(pooled over both texts), gold `-1` dropped, short predictions zero-padded / long
ones truncated — identical to the benchmark's `evaluate_predictions_admin` harness.
Kendall τ-b is reported as the tie-robustness companion.

### 6.1 Intermediate-layer ablation vs. human agreement

Before training the probe we need to know *which layer* carries the cross-lingual
difference signal. `ablation.py` computes a **training-free, closed-form** readout
for every one of Apertus's 32 blocks: an OLS direction (`d ∝ Cov(x, y)`) for the
continuous label (`ρ (OLS)`) and a difference-of-means direction for the binary
same/different split (`ρ (DoM)`, `AUC`, `Sep`). Run on a **dev-only** calibration
slice (`--split val`, 60 pairs; 21 calib / 39 eval); the `test` split is never
touched. ★ = best `ρ (OLS)`, ◆ = best `AUC (DoM)`.

| Layer | Heuristic name | ρ (OLS) | Pearson r | ρ (DoM) | AUC (DoM) | Sep | ρ de | ρ fr | ρ it | n |
| --: | --- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 0 | Surface & orthography | 0.0891 | 0.0951 | 0.0860 | 0.5906 | 0.313 | 0.0662 | 0.0931 | 0.0714 | 26630 |
| 1 | Surface & orthography | 0.0755 | 0.0884 | 0.0694 | 0.5840 | 0.294 | 0.0516 | 0.0615 | 0.0759 | 26630 |
| 2 | Surface & orthography | 0.0612 | 0.0738 | 0.0575 | 0.5759 | 0.256 | 0.0289 | 0.0587 | 0.0634 | 26630 |
| 3 | Surface & orthography | 0.0646 | 0.0813 | 0.0608 | 0.5806 | 0.287 | 0.0328 | 0.0698 | 0.0600 | 26630 |
| 4 | Local syntax / morphology | 0.0665 | 0.0838 | 0.0783 | 0.5949 | 0.341 | 0.0442 | 0.0865 | 0.0480 | 26630 |
| 5 | Local syntax / morphology | 0.0915 | 0.1069 | 0.0982 | 0.6175 | 0.417 | 0.0811 | 0.0956 | 0.0763 | 26630 |
| 6 | Local syntax / morphology | 0.0982 | 0.1237 | 0.0993 | 0.6245 | 0.451 | 0.0857 | 0.0990 | 0.0880 | 26630 |
| 7 | Local syntax / morphology | 0.1288 | 0.1705 | 0.1357 | 0.6627 | 0.599 | 0.1109 | 0.1211 | 0.1351 | 26630 |
| 8 | Local syntax / morphology | 0.1661 | 0.2231 | 0.1674 | 0.6962 | 0.727 | 0.1363 | 0.1474 | 0.1954 | 26630 |
| 9 | Local syntax / morphology | 0.1990 | 0.2458 | 0.2117 | 0.7334 | 0.844 | 0.1643 | 0.1838 | 0.2290 | 26630 |
| 10 | Semantic build-up | 0.2107 | 0.2880 | 0.2239 | 0.7495 | 0.972 | 0.1679 | 0.1880 | 0.2476 | 26630 |
| 11 | Semantic build-up | 0.2129 | 0.3034 | 0.2284 | 0.7506 | 1.019 | 0.1633 | 0.1890 | 0.2556 | 26630 |
| 12 | Semantic build-up | 0.2136 | 0.3441 | 0.2293 | 0.7521 | 1.132 | 0.1694 | 0.1919 | 0.2497 | 26630 |
| 13 | Semantic build-up | 0.2157 | 0.3325 | 0.2368 | 0.7528 | 1.111 | 0.1645 | 0.2033 | 0.2512 | 26630 |
| 14 ◆ | Semantic build-up | 0.2317 | 0.3498 | 0.2571 | **0.7664** | 1.181 | 0.1860 | 0.2254 | 0.2582 | 26630 |
| 15 ★ | Semantic build-up | **0.2318** | **0.3688** | 0.2548 | 0.7612 | **1.206** | 0.1939 | 0.2279 | 0.2497 | 26630 |
| 16 | Mid-depth semantic core | 0.2075 | 0.3284 | 0.2338 | 0.7398 | 1.094 | 0.1745 | 0.2133 | 0.2144 | 26630 |
| 17 | Cross-lingual semantics | 0.2103 | 0.3262 | 0.2374 | 0.7386 | 1.083 | 0.1702 | 0.2156 | 0.2169 | 26630 |
| 18 | Cross-lingual semantics | 0.1968 | 0.2973 | 0.2252 | 0.7235 | 1.006 | 0.1603 | 0.2122 | 0.1933 | 26630 |
| 19 | Cross-lingual semantics | 0.2068 | 0.2980 | 0.2328 | 0.7296 | 1.001 | 0.1692 | 0.2190 | 0.2079 | 26630 |
| 20 | Cross-lingual semantics | 0.1949 | 0.2871 | 0.2237 | 0.7220 | 0.987 | 0.1613 | 0.2044 | 0.1940 | 26630 |
| 21 | Cross-lingual semantics | 0.1969 | 0.2864 | 0.2284 | 0.7302 | 1.006 | 0.1661 | 0.2027 | 0.1964 | 26630 |
| 22 | Task & discourse | 0.1700 | 0.2264 | 0.1934 | 0.6883 | 0.769 | 0.1479 | 0.1815 | 0.1601 | 26630 |
| 23 | Task & discourse | 0.1694 | 0.2202 | 0.1882 | 0.6843 | 0.739 | 0.1497 | 0.1743 | 0.1610 | 26630 |
| 24 | Task & discourse | 0.1582 | 0.2000 | 0.1757 | 0.6735 | 0.684 | 0.1386 | 0.1632 | 0.1448 | 26630 |
| 25 | Task & discourse | 0.1633 | 0.2152 | 0.1823 | 0.6839 | 0.751 | 0.1424 | 0.1676 | 0.1476 | 26630 |
| 26 | Task & discourse | 0.1600 | 0.2214 | 0.1755 | 0.6834 | 0.762 | 0.1441 | 0.1608 | 0.1413 | 26630 |
| 27 | Task & discourse | 0.1487 | 0.2101 | 0.1598 | 0.6687 | 0.693 | 0.1347 | 0.1581 | 0.1298 | 26630 |
| 28 | Task & discourse | 0.1449 | 0.1986 | 0.1558 | 0.6626 | 0.663 | 0.1339 | 0.1605 | 0.1248 | 26630 |
| 29 | Task & discourse | 0.1323 | 0.1726 | 0.1400 | 0.6460 | 0.578 | 0.1188 | 0.1486 | 0.1115 | 26630 |
| 30 | Output formation | 0.1301 | 0.1718 | 0.1364 | 0.6434 | 0.567 | 0.1184 | 0.1533 | 0.1128 | 26630 |
| 31 | Output formation | 0.0724 | 0.0986 | 0.0724 | 0.5890 | 0.331 | 0.0594 | 0.0586 | 0.0592 | 26630 |

**Against human agreement.** The signal rises monotonically through the syntax
band, peaks in the *Semantic build-up* band, and decays toward the output layers.
The strongest single block is **layer 15** (ρ = **0.2318**), with **AUC peaking one
layer earlier at layer 14** (0.7664); the plateau spans **L14–17** — exactly the
band the trained probe consumes. Compared with the human-annotator ceiling
(token-level Spearman ×100):

| | EN–DE | EN–FR | EN–IT |
| --- | --: | --: | --: |
| Human agreement (ceiling) | 56.14 | 43.61 | 65.09 |
| Layer 15 ρ (OLS) ×100 | 19.4 | 22.8 | 25.0 |
| Layer 15 as % of ceiling | 34.5% | 52.3% | 38.4% |

So a single closed-form direction on frozen L15 already recovers roughly a third to
half of the human ceiling — enough signal to justify training the parametric probe,
which then roughly doubles it (§2).

## 7. Limitations

- **French is annotation-limited.** The human ceiling for EN–FR (43.6) is far lower
  than EN–DE (56.1) / EN–IT (65.1), so all systems' FR numbers are compressed.
- **Ablation slice is small** (60 pairs, 21/39 calib/eval); it selects a *band*, not
  a precise layer. It never touches `test`, so the layer choice is not test-tuned.
- **iSTS numbers are small-set** (100/lang) and the `projected` labels are the
  authors' GPT-4o-mini projections, not human labels.
- **x86-64 is slow** (~3.8 tok/s decode under emulation); the engine targets ARM
  SIMD. Long documents dominate runtime.
- The probe is a single seed; the GRU/CNN architecture choice was a near-tie on
  `dev`.

## 8. Reproducibility

- **One command:** `make run` from the repository root (pulls the published
  multi-arch image and prints the bundled results). `make run MODE=predict …`
  recomputes predictions; `make run MODE=tui` opens the browser.
- **Environment:** `LLM_NAME` / `LLM_BASE_URL` / `LLM_API_KEY` are accepted
  (informational — the model is local and embedded).
- **Hardware:** any 64-bit CPU with ~8 GB free RAM. Apple Silicon ≈ 16 tok/s decode
  / 63 tok/s prefill; `--speedtest` reports live numbers.
- **Determinism:** DiffAlign is training-free; the probe is a single seeded
  checkpoint (`track_2a/src/python/layerwise/pairprobe_best_L14-15-16-17_sum_bce_gru_devsel.pth`,
  `dev`-selected). Training/ablation/judging scripts are in `track_2a/src/python/`
  with pinned `requirements.txt`.
- **Build:** `make build` (host arch) or `make push` (multi-arch) from the root
  `Dockerfile`; the GGUF is fetched from Hugging Face at build time.

## 9. Next steps

- Multi-seed ensembling and a FR-focused objective to lift the annotation-limited
  language pair.
- A learned cross-attention head over L14–17 to replace the concatenate-and-GRU
  approximation.
- Distilling the probe into the engine for a single-pass, lower-latency run, and a
  larger long-document evaluation.

## License

Creative Commons Attribution 4.0 (CC-BY-4.0). Code under Apache-2.0. All
HackApertus projects are open-sourced.

## References

1. M. Wastl, J. Vamvas, R. Sennrich. *SwissGov-RSD: A Human-annotated, Cross-lingual Benchmark for Token-level Recognition of Semantic Differences Between Related Documents.* ACL 2026. arXiv:2512.07538.
2. J. Vamvas, R. Sennrich. *Simple and Effective Text Alignment for Machine Translation Evaluation.* 2023.
3. Swiss AI Initiative. *Apertus.* 2026. https://huggingface.co/swiss-ai/Apertus-v1.5-8B
