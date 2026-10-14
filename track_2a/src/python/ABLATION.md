# Apertus layer ablation — which layer is the strongest signal for Semantic Diff

In order to train a probe, we need an initial strong signal (versus ground truth)
This can only be computed by measuring intermediate layers for correlations
Once the intermediate layer is decided, we train an actual parametric probe 
This forms a nonlinear scoring mechanism for individual and overall tokens from different languages.
Based on whether they are semantically similar or different to each other.

## Spearman Correlations using different techniques (from the paper)

Wastl, Vamvas & Sennrich (ACL 2026), Table 3 — SwissGov-RSD columns. Token-level Spearman
with gold labels (values as reported in the paper, ×100). ★ = best overall on SwissGov-RSD.
`DE/FR/IT %` = each score as a percentage of that language's human-annotator agreement
(e.g. `20.6 / 56.1 = 36.7%`); the best ratio is bolded.

| Technique                      | Type              | EN–DE    | EN–FR    | EN–IT    | DE %  | FR %  | IT %     |
|--------------------------------|-------------------|--:-------|--:-------|--:-------|--:----|--:----|--:-------|
| DiffAlign · LaBSE ★            | unsupervised      | 20.6     | 8.2      | **25.6** | 36.7  | 18.8  | **39.3** |
| DiffAlign · XLM-R+SimCSE       | unsupervised      | 19.2     | 10.0     | 23.4     | 34.2  | 22.9  | 35.9     |
| DiffAlign · Qwen3-Embedding 4B | unsupervised      | 20.1     | 11.2     | 20.3     | 35.8  | 25.7  | 31.2     |
| DiffAlign · EuroBERT 610M      | unsupervised      | 18.6     | 11.4     | 18.7     | 33.2  | 26.1  | 28.7     |
| DiffAlign · bge-m3             | unsupervised      | 14.8     | 9.9      | 13.6     | 26.4  | 22.7  | 20.9     |
| GPT-4o                         | few-shot          | 4.2      | 1.2      | 5.5      | 7.5   | 2.8   | 8.4      |
| GPT-4o-mini                    | fine-tuned (iSTS)  | 3.9      | 1.1      | 5.3      | 7.0   | 2.5   | 8.1      |
| ModernBERT-large (EN-IT*)      | fine-tuned (iSTS)  | 7.5      | 4.8      | 12.4     | 13.4  | 11.0  | 19.0     |
| **Human annotator agreement**  | practical ceiling | **56.1** | **43.6** | **65.1** | 100.0 | 100.0 | 100.0    |


**Best on SwissGov-RSD: 25.6 (LaBSE DiffAlign, EN–IT).** The paper's point is the
*generalization gap*: the same systems score ~45–82 on the synthetic iSTS-RSD but only
~0–26 here. Note all paper numbers on SwissGov-RSD are **zero-shot / transferred from
iSTS-RSD** (DiffAlign is unsupervised; the fine-tuned encoders/LLMs train on synthetic data,
not on SwissGov), so our in-domain probe (see the ablation table below) is not directly
comparable — the fair reference is the human ceiling ratio that we computed.


## The 32 blocks, with friendly (heuristic) names

> The names below are a *general* mental model of how decoder LLMs spread work
> (surface → syntax → semantics → task → output). They are **not** measured specifically for
> Apertus; the ablation table is what turns them into evidence.

```mermaid
flowchart LR
  subgraph L0["Blocks 0–3 · SURFACE & ORTHOGRAPHY"]
    A0["0–3<br/>casing, spelling,<br/>subword / n-gram identity"]
  end
  subgraph L1["Blocks 4–9 · LOCAL SYNTAX / MORPHOLOGY"]
    A1["4–9<br/>POS, agreement,<br/>short phrases"]
  end
  subgraph L2["Blocks 10–15 · SEMANTIC BUILD-UP"]
    A2["10–15<br/>lexical meaning,<br/>composition"]
  end
  subgraph L3["Block 16 · MID-DEPTH SEMANTIC CORE"]
    A3["16<br/>chosen by the notebook<br/>(≈ 50% depth)"]
  end
  subgraph L4["Blocks 17–21 · CROSS-LINGUAL SEMANTICS"]
    A4["17–21<br/>paraphrase / translation<br/>invariance, abstract meaning"]
  end
  subgraph L5["Blocks 22–29 · TASK & DISCOURSE"]
    A5["22–29<br/>intent, coherence,<br/>what to say next"]
  end
  subgraph L6["Blocks 30–31 · OUTPUT FORMATION"]
    A6["30–31<br/>next-token / logit prep"]
  end
  A0 --> A1 --> A2 --> A3 --> A4 --> A5 --> A6
```

NOTE: Apertus is a dense transformer, not an MoE, and every layer encodes information
about the input tokens, to be used in generating the output. Effectively we distill intermediate
representations of the model's multilingual understanding to ensure an effective probe.

## Ablation Method (no probe training)

`ablation.py` computes, for every block, a **closed-form / training-free** readout
of how well the block's features encode the human token-level difference labels
(all languages: de + fr + it, ground truth `labels_a`/`labels_b`):

1. Capture site `res_ffn` for **all 32 blocks in a single prefill** (per side), for
   each pair, and mean-pool token vectors to whitespace words.
2. Estimate two directions on a **calibration slice** (`--calib-frac`, default 0.35):
   * **OLS** — `d ∝ Cov(x, y)` (closed-form least squares; the "probe without the
     Transformer"), aimed at the **continuous** label → reported `Spearman ρ (OLS)`.
   * **DoM** — difference of class means `μ(gold≥0.5) − μ(gold≤0.05)`, aimed at the
     **binary** same/different split → reported `ρ (DoM)`, `AUC`, `Sep`.
3. Score the held-out pairs as `s = x·d` and report per block:
   * `Spearman ρ (OLS)` — rank correlation with the gold score (the metric the real
     probe is judged on) → **primary**.
   * `Pearson r (OLS)`, `ρ (DoM)`, `AUC (DoM)` (same vs different), `Sep`
     (standardised projection gap), and per-language `ρ de/fr/it`.

## Word Align method and Innovation for final classification

Two inferences of Apertus per pair. Because Apertus is causal — a token only sees what's *before* it — you can't get both cross-contexts in one pass.

- **Pass 1 = `[B][A]`** (German then English) → English tokens sit *after* German, so they attend to it → keep the English rows = `feats_a` (English sees German).
- **Pass 2 = `[A][B]`** (English then German) → German tokens sit *after* English → keep the German rows = `feats_b` (German sees English).

In `[A][B]` alone the English tokens are first, so they'd only see themselves (useless); the German tokens would see English. In `[B][A]` it's the reverse. So you need **both orders** to get each side's vector with the other language in context.

> Cost note: that is 2× a single prefill. We added a capture-only early exit (`model_capture_stop`) that breaks the layer loop at L15 → ~2× faster capture while producing **bit-identical** L15 vectors. Text generation is unaffected.

### For Example
One EN–DE pair (`admin_de_1`):

```
Apertus (frozen):     feats_a (305, 4096)   [English, sees German]
                      feats_b (250, 4096)   [German,  sees English]
concat:               X      (555, 4096)    # English 0..304, German 305..554
seg:                  seg    (555,)         # 305 zeros, then 250 ones (which text)
pad:                  pad    (555,)  bool   # padding mask (batch)
                      |
proj (4096->512):     (555, 512)
+ seg embed:          (555, 512)
transformer (4L,8H):  (555, 512)            # attention across BOTH texts
head (512->128->1):   (555, 1) -> (555,)
sigmoid:              (555,)                # one 0-1 score per token
```

Output = 555 scores: `[0:305]` English tokens, `[305:555]` German tokens. `seg`/`pad` are bookkeeping we know from how the pair was built — nothing predicted, nothing from labels.

## Simplified Understanding

The idea in one picture:

```
English text ──Apertus (frozen)──► per-token vectors  (n_EN, 4096)
German  text ──Apertus (frozen)──► per-token vectors  (n_DE, 4096)
                      |  concatenate [English ; German]  + seg marker (0/1)
                      v
              small transformer  (cross-attention over both texts)
                      |
              per-token head  ──►  one 0–1 difference score per token
```

- **Apertus stays frozen**; only the small transformer ("pair probe") is trained.
- Because both texts sit in **one sequence**, every German token can attend to the English tokens (and vice versa) — so the model can *compare* the two languages. Scoring each side separately cannot do this; that is the whole point of the pair probe.


## Current results

SwissGov-RSD **test** split (56/lang), official judging metric (global Spearman
×100, gold `-1` dropped).

| method                                                  | EN–DE      | EN–FR      | EN–IT      |
|---------------------------------------------------------|--:---------|--:---------|--:---------|
| DiffAlign · Apertus L15 (unsupervised)                  | 32.2       | 18.2       | 36.0       |
| DiffAlign · Apertus L14–17 sum (unsupervised)           | 31.5       | 18.4       | 36.6       |
| **Pair probe · GRU (frozen Apertus L14–17, dev-selected)** | **44.1** | **27.2** | **47.3** |
| Human annotator agreement (ceiling)                     | 56.1       | 43.6       | 65.1       |


*Values are token-level Spearman ×100, pooled over both texts, **test** split.
The probe is a **bidirectional GRU** (final arch), single seed, best **dev** epoch
(ep5); trained and early-stopped on `dev` only — `test` is never used for
selection. An earlier `46.2 / 27.3 / 50.1` run selected on the test split and is
superseded (the ~+2 gap was that optimism).*

*Architecture (dev-selected, single seed). A 1-epoch dev screen (ep0 macro) —
**CNN 0.3624**, GRU 0.3506, bilinear 0.3438, MLP 0.3360, transformer 0.3343,
xattn 0.1631 — flagged CNN as the only close contender, so it was run to
completion: CNN peaks at **0.38322** (ep4) vs GRU **0.38323** (ep5) — a **tie**
(GRU ahead on IT 0.448 vs 0.444 and FR 0.265 vs 0.263; CNN ahead on DE 0.443 vs
0.437). GRU is **retained**. FR is annotation-limited (ceiling 43.6). DiffAlign
L14–17 fusion does **not** beat L15 on SwissGov (28.8 vs 28.8). The synthetic
target↔target OOD table is **exploratory only** (built from the test split) and is
excluded from judged claims.*



## Ablation table for intermediate Layers compared against Human Agreement

Run (dev-selected): `python ablation.py --split val --limit 20 --calib-frac 0.35 --out ablation_devsel.json`
(60 pairs: 21 calib / 39 eval; ★ = best `Spearman ρ (OLS)`, ◆ = best `AUC (DoM)`).
Never touches the test split.

**Result:** the signal is strongest at **layer 15** (ρ = **0.2318**, AUC = 0.7612)
— the top of the *Semantic build-up* band — with AUC peaking one layer earlier at
**layer 14** (0.7664); the plateau spans roughly **layers 14–17**, confirming the
L14–17 probe band from `dev` alone. (Numbers are lower than the earlier
`--split full` run because that run included the test split; those are superseded.)

| Layer  | Heuristic name            | Spearman ρ (OLS) | Pearson r (OLS) | ρ (DoM) | AUC (DoM) | Sep   | ρ de   | ρ fr   | ρ it   | n     |
|--:-----|---------------------------|--:---------------|--:--------------|--:------|--:--------|--:----|--:-----|--:-----|--:-----|--:----|
| 0      | Surface & orthography     | 0.0891           | 0.0951          | 0.0860  | 0.5906    | 0.313 | 0.0662 | 0.0931 | 0.0714 | 26630 |
| 1      | Surface & orthography     | 0.0755           | 0.0884          | 0.0694  | 0.5840    | 0.294 | 0.0516 | 0.0615 | 0.0759 | 26630 |
| 2      | Surface & orthography     | 0.0612           | 0.0738          | 0.0575  | 0.5759    | 0.256 | 0.0289 | 0.0587 | 0.0634 | 26630 |
| 3      | Surface & orthography     | 0.0646           | 0.0813          | 0.0608  | 0.5806    | 0.287 | 0.0328 | 0.0698 | 0.0600 | 26630 |
| 4      | Local syntax / morphology | 0.0665           | 0.0838          | 0.0783  | 0.5949    | 0.341 | 0.0442 | 0.0865 | 0.0480 | 26630 |
| 5      | Local syntax / morphology | 0.0915           | 0.1069          | 0.0982  | 0.6175    | 0.417 | 0.0811 | 0.0956 | 0.0763 | 26630 |
| 6      | Local syntax / morphology | 0.0982           | 0.1237          | 0.0993  | 0.6245    | 0.451 | 0.0857 | 0.0990 | 0.0880 | 26630 |
| 7      | Local syntax / morphology | 0.1288           | 0.1705          | 0.1357  | 0.6627    | 0.599 | 0.1109 | 0.1211 | 0.1351 | 26630 |
| 8      | Local syntax / morphology | 0.1661           | 0.2231          | 0.1674  | 0.6962    | 0.727 | 0.1363 | 0.1474 | 0.1954 | 26630 |
| 9      | Local syntax / morphology | 0.1990           | 0.2458          | 0.2117  | 0.7334    | 0.844 | 0.1643 | 0.1838 | 0.2290 | 26630 |
| 10     | Semantic build-up         | 0.2107           | 0.2880          | 0.2239  | 0.7495    | 0.972 | 0.1679 | 0.1880 | 0.2476 | 26630 |
| 11     | Semantic build-up         | 0.2129           | 0.3034          | 0.2284  | 0.7506    | 1.019 | 0.1633 | 0.1890 | 0.2556 | 26630 |
| 12     | Semantic build-up         | 0.2136           | 0.3441          | 0.2293  | 0.7521    | 1.132 | 0.1694 | 0.1919 | 0.2497 | 26630 |
| 13     | Semantic build-up         | 0.2157           | 0.3325          | 0.2368  | 0.7528    | 1.111 | 0.1645 | 0.2033 | 0.2512 | 26630 |
| 14 ◆   | Semantic build-up         | 0.2317           | 0.3498          | 0.2571  | 0.7664    | 1.181 | 0.1860 | 0.2254 | 0.2582 | 26630 |
| 15 ★   | Semantic build-up         | 0.2318           | 0.3688          | 0.2548  | 0.7612    | 1.206 | 0.1939 | 0.2279 | 0.2497 | 26630 |
| 16     | Mid-depth semantic core   | 0.2075           | 0.3284          | 0.2338  | 0.7398    | 1.094 | 0.1745 | 0.2133 | 0.2144 | 26630 |
| 17     | Cross-lingual semantics   | 0.2103           | 0.3262          | 0.2374  | 0.7386    | 1.083 | 0.1702 | 0.2156 | 0.2169 | 26630 |
| 18     | Cross-lingual semantics   | 0.1968           | 0.2973          | 0.2252  | 0.7235    | 1.006 | 0.1603 | 0.2122 | 0.1933 | 26630 |
| 19     | Cross-lingual semantics   | 0.2068           | 0.2980          | 0.2328  | 0.7296    | 1.001 | 0.1692 | 0.2190 | 0.2079 | 26630 |
| 20     | Cross-lingual semantics   | 0.1949           | 0.2871          | 0.2237  | 0.7220    | 0.987 | 0.1613 | 0.2044 | 0.1940 | 26630 |
| 21     | Cross-lingual semantics   | 0.1969           | 0.2864          | 0.2284  | 0.7302    | 1.006 | 0.1661 | 0.2027 | 0.1964 | 26630 |
| 22     | Task & discourse          | 0.1700           | 0.2264          | 0.1934  | 0.6883    | 0.769 | 0.1479 | 0.1815 | 0.1601 | 26630 |
| 23     | Task & discourse          | 0.1694           | 0.2202          | 0.1882  | 0.6843    | 0.739 | 0.1497 | 0.1743 | 0.1610 | 26630 |
| 24     | Task & discourse          | 0.1582           | 0.2000          | 0.1757  | 0.6735    | 0.684 | 0.1386 | 0.1632 | 0.1448 | 26630 |
| 25     | Task & discourse          | 0.1633           | 0.2152          | 0.1823  | 0.6839    | 0.751 | 0.1424 | 0.1676 | 0.1476 | 26630 |
| 26     | Task & discourse          | 0.1600           | 0.2214          | 0.1755  | 0.6834    | 0.762 | 0.1441 | 0.1608 | 0.1413 | 26630 |
| 27     | Task & discourse          | 0.1487           | 0.2101          | 0.1598  | 0.6687    | 0.693 | 0.1347 | 0.1581 | 0.1298 | 26630 |
| 28     | Task & discourse          | 0.1449           | 0.1986          | 0.1558  | 0.6626    | 0.663 | 0.1339 | 0.1605 | 0.1248 | 26630 |
| 29     | Task & discourse          | 0.1323           | 0.1726          | 0.1400  | 0.6460    | 0.578 | 0.1188 | 0.1486 | 0.1115 | 26630 |
| 30     | Output formation          | 0.1301           | 0.1718          | 0.1364  | 0.6434    | 0.567 | 0.1184 | 0.1533 | 0.1128 | 26630 |
| 31     | Output formation          | 0.0724           | 0.0986          | 0.0724  | 0.5890    | 0.331 | 0.0594 | 0.0586 | 0.0592 | 26630 |


## How to read it

* We look for where `Spearman ρ (OLS)` (and `AUC`) peaks like a local minima — that band is
  the candidate region for the real probe.
* Cross-lingual stability matters for RSD: we prefer a band where `ρ de`, `ρ fr`,
  `ρ it` agree in sign and magnitude rather than one language dominating.
* The chosen `LAYER` is one that makes the probe's life easier, i.e. by giving a clear
  signal to the parametric classifier we ensure a distinct output as well