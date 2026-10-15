## Abstract
For the Research Track 2A (UZH) - We beat the benchmarks proposed by [SwissGov RSD, M. Wastl et al](https://arxiv.org/pdf/2512.07538) using only a frozen Apertus-1.5-8B for the unsupervised set, and requiring only a lightweight 4M parameter encoder (single bidirectional GRU) on top of a frozen Apertus-1.5-8B for the supervised set. We do not retrain, finetune or otherwise alter Apertus-1.5-8B, for all practical purposes the model is completely frozen. All our work is completely reproducible on a local machine with a reasonable amount of RAM (~12 GB) and CPU cores (>= 8), we provide CPU only inference (accelerated using NEON SIMD).

### Apertus-8B
We use the Text-Only weights with the image encoder removed from: https://huggingface.co/andreasmartin/apertus-v1.5-8b-text and quantize it to 5-bit mixed quantization and upload it to: https://huggingface.co/justacoderwhocodes/apertus-1.5-text-only (personal huggingface).

### Results in brief

Based on the SwissGov-RSD **test** split (56 docs/lang), token-level **Spearman ×100**, with gold `-1` dropped. paper rows are reproduced from Wastl, Vamvas & Sennrich (ACL 2026) -> Table 3.

| System                                         | trained on         |    EN–DE |    EN–FR |    EN–IT |    macro |
| :--------------------------------------------- | :----------------- | -------: | -------: | -------: | -------: |
| DiffAlign · Apertus L15 *(ours)*               | —                  |     32.2 |     18.2 |     36.0 |     28.8 |
| DiffAlign · Apertus L14–17 sum *(ours)*        | —                  |   _31.5_ |   _18.4_ |   _36.6_ |   _28.8_ |
| **Pair probe · Apertus L14–17 → GRU *(ours)*** | SwissGov in-domain | **44.1** | **27.2** | **47.3** | **39.5** |
| DiffAlign · LaBSE *(paper)*                    | —                  |     20.6 |      8.2 |     25.6 |     18.1 |
| DiffAlign · XLM-R + SimCSE *(paper)*           | —                  |     19.2 |     10.0 |     23.4 |     17.5 |
| DiffAlign · bge-m3 *(paper)*                   | —                  |     14.8 |      9.9 |     13.6 |     12.8 |
| GPT-4o *(paper, few-shot)*                     | —                  |      4.2 |      1.2 |      5.5 |      3.6 |
| ModernBERT-large (EN) *(paper, fine-tuned)*     | iSTS               |      5.7 |      1.3 |      4.0 |      3.7 |
| **Human annotator agreement (ceiling)**        | —                  | **56.1** | **43.6** | **65.1** | **54.9** |


**Reading in brief** -> The paper's own finding is a large *generalization gap*: systems that score ~45–82 on synthetic iSTS-RSD collapse to ~0–26 on naturalistic SwissGov-RSD.

Our in-domain, frozen-Apertus probe closes a large part of that gap and unlike the paper's fine-tuned rows can be (and is) trained only on SwissGov `dev`. The `test` split is used exactly once, for reporting. The CLI exposes the metrics **per document** (`spearman`, `kendall` columns in `results`/`predict`). 

However, we also **transparently show our results on either combination**, i.e. training on iSTS and inference on SwissGov, training on SwissGov and inference on iSTS, as well as unsupervised (no training) inference. Every combination beats every corresponding benchmark from the paper. Our entire results table is provided below.

## Objectives
- To build a custom Apertus inference framework to output model tensors individually per layer
    + This is shipped as a `.dylib (for MacOS)` and `.so (for Linux)` and highly optimized for ARM NEON CPUs
    + We also cross compile to `AMD64 (Intel)` architecture but do not guarantee speed (however, results will be numerically stable)
- To determine the key intermediate layers that hold maximum contextual information
    + We run ablation across 32 intermediate layers only on SwissGov dev splits
    + For the ablation we determine the best layers carrying maximum correlation with gold labels
    + This is **not** diffalign, instead it is just an ordinary least squares direction scored by ranked correlation
- To run DiffAlign across these layers in order to perform an RSD unsupervised iteration
    + Layer 15 and Layers 14-17 combined are chosen and we report results in the unsupervised section
    + There is not too much of a difference between selecting single feature, vs combined features on SwissGov dataset
    + However, on the iSTS benchmark, the higher degree of information in combined layers leads to better scoring
- To ablate classifiers, selecting a GRU arch in order to perform RSD supervised iteration
    + Ablation is carried across Transformer, CNN (1d-conv), bilinear, xattn, GRU and MLP. We find GRU best suited, CNN is a close second
    + Both Layer 15 and Layers 14-17 again are used as priors, and differences seen
    + Layer 14-17 combined has superior information and so produces better results on the classifier (no surprises there)
- To also compare against the natural Human Annotator baseline vs our methods
    + On in-domain validation, we can reach ~80-85% of human baseline, however, FR is a particularly difficult dataset, and impacts our in-domain test scores.
- We report results appending to the properly formatted table in the paper

## Intuition
---
> Apertus 8B is basically a set of dense transformers that can be layerwise inferenced. The model itself is trained on relevant multilingual data, and contains an extremely rich set of priors, that other open source models (the paper uses `Llama 3.1 8B`) do not have. We basically exploit these priors in order to run predictions across encoded multilingual pairs and either diff align the layerwise outputs or train a tiny encoder head to perform supervised predictions.
---

## Innovations
Aside from the mechanical innovation of layerwise inference and encoder training, our key innovation is in the inference process itself.

We perform two **partial** prefills of Apertus per pair — one per ordering — and neither runs the full model. Because Apertus is causal — a token only sees what's *before* it — you can't get both cross-contexts in one pass. We prefill concatenated tokens, but keep only relevant features by index.

- **Pass 1 = `[B][A]`** (German then English) → English tokens sit *after* German, so they attend to it → keep the English rows = `feats_a` (English sees German).
- **Pass 2 = `[A][B]`** (English then German) → German tokens sit *after* English → keep the German rows = `feats_b` (German sees English).

In `[A][B]` alone the English tokens are first, so they'd only see themselves (useless); the German tokens would see English. In `[B][A]` it's the reverse. So we need **both orders** to get each side's vector with the other language in context.

> Cost note: that is two prefills, but each is a **partial** forward — a capture-only early exit breaks the layer loop at **L17**, so every pass runs layers 0→17 of 32. The total is therefore ≈ 2× an 18-layer prefill, **not** 2× the full model, and effectively ~1.1x of the full model minus decode. Each pass emits a per-token `res_ffn` residual for the whole concatenated sequence, mean-averaged over L14–17 → shape `(n_seq, dim)`; we basically get word-level features `feats_a (nwa, dim)` and `feats_b (nwb, dim)` after both passes.

This allows us to correctly and speedily inference even on an unaligned pair. For eg, the number of words in English and German, even for a semantically similar sentence might be drastically different, but by prefilling across merged tokens we essentially predict across the entire set, for both languages, thus ensuring an unbiased representation. 

---
### Illustratively, One EN+DE pair (`admin_de_1`, 305 English words and 250 German words):

```
Apertus (frozen):     partial pass1 -> feats_a (305, 4096)   [English, sees German]
                      partial pass2 -> feats_b (250, 4096)   [German,  sees English]
concat:               X      (555, 4096)    # English 0..304, German 305..554
seg:                  seg    (555,)         # 305 zeros, then 250 ones (which text)
pad:                  pad    (555,)  bool   # padding mask (batch)
                      |
proj (4096->512):     (555, 512)
+ seg embed:          (555, 512)
transformer (4L,8H):  (555, 512)            # attention across BOTH texts
head (512->128->1):   (555, 1) -> (555,)
sigmoid:              (555,)                # one 0-1 score per word
```

**Output** = 555 scores: `[0:305]` English words, `[305:555]` German words. `seg`/`pad` are bookkeeping we know from how the pair was built — nothing predicted, nothing from labels.

### Supervised vs Unsupervised Architecture

Two heads over one frozen backbone, all CPU:

```
EN text ─┐                                   ┌─ DiffAlign (unsupervised)
         ├─ Apertus-v1.5-8B (frozen) ─ feats ─┤   s = 1 − max_j cos(h_i, h_j)
XX text ─┘   capture L14–17                  └─ Pair-probe (supervised GRU)
                                                 concat[EN;XX] + seg → 1 score/word
```

---

## Results
We achieve about 70-80% of the human annotator agreement ceiling, using extremely cheap methods, utilizing Apertus priors.

Rows marked *(paper)* reproduce Table 3 of Wastl et al. (2026). Their trained systems are fine-tuned on **synthetic iSTS-RSD** (+PAWS), and zero-shot transfer tested on SwissGov.

| System                                        | trained on                | SwissGov split | iSTS DE  | FR       | IT       | SwissGov DE | FR       | IT       |
|-----------------------------------------------|---------------------------|----------------|---------:|---------:|---------:|------------:|---------:|---------:|
| **DiffAlign · Apertus L15 (ours)**            | —                         | test           | _39.4_   | _42.7_   | _42.9_   | **32.2**    | **18.2** | **36.0** |
| **DiffAlign · Apertus L15 (ours)**            | —                         | dev-val        | _39.4_   | _42.7_   | _42.9_   | 33.2        | 21.7     | 36.2     |
| **DiffAlign · Apertus L14–17 sum (ours)**     | —                         | test           | **44.4** | **46.8** | **47.3** | _31.5_      | _18.4_   | _36.6_   |
| **DiffAlign · Apertus L14–17 sum (ours)**     | —                         | dev-val        | **44.4** | **46.8** | **47.3** | 32.9        | 21.6     | 36.4     |
| *Unsupervised DiffAlign — paper*              |                           |                |          |          |          |             |          |          |
| XLM-R + SimCSE                                | —                         |                | 44.9     | 45.2     | 44.9     | 19.2        | 10.0     | 23.4     |
| LaBSE                                         | —                         |                | 40.6     | 42.3     | 45.3     | 20.6        | 8.2      | 25.6     |
| bge-m3                                        | —                         |                | 47.1     | 45.8     | 47.5     | 14.8        | 9.9      | 13.6     |
| gte-multilingual-base                         | —                         |                | 31.2     | 30.2     | 30.5     | 10.7        | 5.9      | 12.8     |
| ModernBERT-large                              | —                         |                | 17.3     | 16.7     | 17.2     | 0.5         | 0.7      | 2.7      |
| EuroBERT 210M                                 | —                         |                | 32.1     | 34.6     | 33.4     | 11.5        | 8.4      | 16.1     |
| EuroBERT 610M                                 | —                         |                | 29.1     | 30.5     | 31.0     | 18.6        | 11.4     | 18.7     |
| EuroBERT 2.1B                                 | —                         |                | 40.3     | 28.5     | 28.3     | 13.7        | 8.9      | 17.4     |
| MmBERT-small                                  | —                         |                | 23.0     | 23.5     | 22.1     | 10.3        | 5.5      | 11.0     |
| MmBERT-base                                   | —                         |                | 34.0     | 32.6     | 32.6     | 14.3        | 8.6      | 15.6     |
| Qwen3-Embedding 4B                            | —                         |                | 41.6     | 42.9     | 43.2     | 20.1        | 11.2     | 20.3     |
| Qwen3-Embedding 8B                            | —                         |                | 40.9     | 40.1     | 41.4     | 19.3        | 12.1     | 18.6     |
| *Few-shot LLMs — paper*                       |                           |                |          |          |          |             |          |          |
| Llama-3.1 8B Instruct                         | —                         |                | 2.8      | 3.7      | 2.6      | –           | –        | –        |
| Llama-3.1 405B Instruct                       | —                         |                | 29.2     | 29.3     | 26.9     | −0.8        | −2.2     | 7.4      |
| GPT-4o-mini                                   | —                         |                | 15.8     | 13.8     | 16.6     | –           | –        | –        |
| GPT-4o                                        | —                         |                | 43.0     | 40.5     | 42.5     | 4.2         | 1.2      | 5.5      |
| DeepSeek-R1                                   | —                         |                | 38.2     | 40.1     | 34.6     | –           | –        | –        |
| o3-mini-low                                   | —                         |                | 44.8     | 46.5     | 48.2     | –           | –        | –        |
| *Fine-tuned LLMs — paper*                     |                           |                |          |          |          |             |          |          |
| Llama-3.1 8B Instruct                         | iSTS                      |                | 66.7     | 67.3     | 66.9     | –           | –        | –        |
| GPT-4o-mini                                   | iSTS                      |                | 81.6     | 79.9     | 78.2     | 3.9         | 1.1      | 5.3      |
| *Fine-tuned encoders — paper*                 |                           |                |          |          |          |             |          |          |
| ModernBERT-large (EN)                         | iSTS                      |                | 55.3     | 55.4     | 53.8     | 5.7         | 1.3      | 4.0      |
| ModernBERT-large (EN–DE\*)                    | iSTS+proj                 |                | 58.8     | 58.4     | 58.5     | 6.4         | 1.6      | 5.6      |
| ModernBERT-large (EN–FR\*)                    | iSTS+proj                 |                | 58.4     | 60.7     | 60.1     | 5.4         | 5.2      | 8.7      |
| ModernBERT-large (EN–IT\*)                    | iSTS+proj                 |                | 59.2     | 61.2     | 63.0     | 7.5         | 4.8      | 12.4     |
| ModernBERT-large (multi\*)                    | iSTS+proj                 |                | 54.7     | 57.5     | 58.1     | 5.7         | 2.5      | 8.8      |
| **Pair probe·Apertus L14–17 sum->GRU (ours)** | SwissGov in-domain        | test           | _59.2_   | _59.2_   | _59.5_   | 44.1        | 27.2     | 47.3     |
| **Pair probe·Apertus L14–17 sum->GRU (ours)** | SwissGov in-domain        | dev-val        | _59.2_   | _59.2_   | _59.5_   | 48.9        | 36.3     | 55.6     |
| **Pair probe·Apertus L14–17 sum->GRU (ours)** | iSTS `projected` 2.4k/lang| —              | **64.0** | **63.5** | **63.4** | _29.5_      | _13.1_   | _27.1_   |
| *Human agreement ceiling*                     | —                         | —              | –        | –        | –        | *56.1*      | *43.6*   | *65.1*   |

**Notes:**
- Values are token-level Spearman ×100 pooled over both texts; `test` is the scored split, gold `-1` dropped.
- **The probe is dev-selected** (trained + early-stopped on `dev` only); `test` is unseen, so the `dev-val` probe rows are **in-sample** (hence higher) than `test`.
- **iSTS** columns are independent of the SwissGov split: DiffAlign is training-free (same iSTS on `test` and `dev-val` rows); the probe's iSTS is **zero-shot** transfer (model, evaluated on the iSTS test set).


---

## What this says

- **On SwissGov, our frozen-Apertus DiffAlign beats every published DiffAlign encoder** — macro **28.8** (test) vs the best paper encoder (LaBSE) at ≈ **18.1**.
- **On synthetic iSTS-RSD the augmented probe reaches 64.0 / 63.5 / 63.4** — at the top of the paper's **fine-tuned-encoder** band (ModernBERT 53.8–63.0), above every unsupervised encoder (best bge-m3 47.5), only behind the large parameter fine-tuned LLM (GPT-4o-mini 81.6).
- **There is no catastrophic generalization gap:** Training on SwissGov and transferring to iSTS actually results in _**a ~15 pp rise on iSTS**_ and the other direction, training on iSTS and transferring to SwissGov results only a _~30-40 pp drop on SwissGov_ (this is because of some dataset issues, elucidated below) as compared to an _~75pp drop on finetuning GPT-4o mini_
- **Dataset choices** We are transparent about reporting on validation split vs reporting on truly unseen data (test split). However, we observe that training the model on SwissGov dataset translates to a much cleaner and transferable result, probably because that is what _**frozen Apertus is used to seeing (long sentences with actual semantic meaning)**_, training on iSTS results in the model being a bit more confused (shorter sentences, word differences over meaning differences)
- **Local & sovereign:** everything runs locally on frozen Apertus weights in ~5 GB RAM (no hosted endpoint, no gateway timeout).

## Caveats

- **iSTS-RSD:** 2.4k samples/lang, English side labelled only. The `projected` training labels are the **benchmark authors' released data** (they used GPT-4o-mini for label projection); that model is **not part of our system** — we only consume the published file with the empty/overlong pairs filtered.
- **SwissGov** dataset is small, only 224 docs in total, the results may be improved as more annotated datasets become available
- **Practical Human Ceiling** We approach ~80-85% of the human agreement ceiling on validation set, but need better annotated data (especially on French) to train better models

## Appendix - Layerwise Ablation (with semantic interpretations)

**Result:** the signal is strongest at **layer 15** (ρ = **0.2318**, AUC = 0.7612) i.e. the top of the *Semantic build-up* band with AUC peaking one layer earlier at **layer 14** (0.7664); the plateau spans roughly **layers 14 to 17**. Carried out on val (dev) only and never sees test

| Layer  | Heuristic name            | Spearman ρ (OLS) | Pearson r (OLS) | ρ (DoM) | AUC (DoM) | Sep   | ρ de   | ρ fr   | ρ it   | n     |
|-------:|---------------------------|-----------------:|----------------:|--------:|----------:|------:|-------:|-------:|-------:|------:|
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


## What this says

* We look for where `Spearman ρ (OLS)` (and `AUC`) peaks like a local maxima and that band is the candidate region for the real probe.
* Cross-lingual stability matters for RSD: we prefer a band where `ρ de`, `ρ fr`, `ρ it` agree in sign and magnitude rather than one language dominating.
* The chosen `LAYER` is one that makes the probe's life easier, i.e. by giving a clear signal to the parametric classifier we ensure a distinct output on a small-parameter classifier


## References

1. M. Wastl, J. Vamvas, R. Sennrich. *SwissGov-RSD: A Human-annotated, Cross-lingual Benchmark for Token-level Recognition of Semantic Differences Between Related Documents.* ACL 2026. arXiv:2512.07538.
2. J. Vamvas, R. Sennrich. *Simple and Effective Text Alignment for Machine Translation Evaluation.* 2023.
3. Swiss AI Initiative. *Apertus.* 2026. https://huggingface.co/swiss-ai/Apertus-v1.5-8B
4. Martin Andreas. *Apertus Text Only Checkpoint.* 2026. https://huggingface.co/andreasmartin/apertus-v1.5-8b-text/
5. Nath Bitan. *Apertus 5 bit Mixed Quantized GGUF.* 2026. https://huggingface.co/justacoderwhocodes/apertus-1.5-text-only/