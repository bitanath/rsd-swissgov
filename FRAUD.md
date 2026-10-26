Track 2A - UZH - Academia -> Your Competition is Attainted by Selecting an objectively inferior project (lower metrics, fudged reporting) as your winner. Are you a FRAUD?

## Hack Apertus the competition appears to be a FRAUD

I would strongly advise against joining any iteration of this competition. There is massive judging bias favoring Swiss Participants. My original Readme can be found at [ORIGINAL_README.md](ORIGINAL_README.md)

# Track 2A: UZH SwissGov RSD challenge 

Winner's repo: https://github.com/amunozo/apertus-swissgov-rsd
My repo: https://github.com/bitanath/rsd-swissgov

My middle-layer ablation and two-prefill design were public on **Oct 11**. The winner’s first solution commit is **Oct 13 10:47** — two days later. The winner's winning number on Test dataset is **37.4 macro**, I lead **39.5 macro**. The winner’s best-case dev/val number on OOF 5 CV is **39.9**; my validation set tests (albeit no cross fold validation) sits at **46.93**. 

I built an actual CPU optimized framework to capture intermediate outputs in a single forward pass. This is not easy, and is a non-trivial coding problem that I solved in Zig (low level systems programming language).
The winner's repo hand-waves at this and uses Huggingface code to hook into intermediate layers which is inefficient and RAM heavy and needs multiple passes, not to mention just maybe 20 lines of code.

I took the time to do an ablation on individual layers to determine signal strength across layers. This idea was copied as is, but the implementation sucked.
The winner's **unsupervised** DiffAlign results actually **maximise Layerwise Diffalign**. Their result passes off **supervised learning** as **unsupervised**.

For the supervised metric, I trained not only on SwissGov RSD, but also on iSTS as a transfer experiment to ensure generalizability. I also trained multiple archs and empirically proved GRU outperformed the rest.
The winner's **supervised** metric simply hand-waves, and does some fancy **tainted** stats, like dev/val crossfold, or like reporting on all training data **to even come close to my results (not beat them)**.

They get a **worse result for the same idea** ... WHY? Because I had wilfully obfuscated my single pass inference code, this allows me a low level look at tensors, and the intuition to build a more optimal framework for inference. This is something **AI Agents cannot do** not yet anyway. I have mailed the organisers my inference code as well, even though this is not required, as proof of work, and to prevent flimsy excuses like not all code was submitted etc. This was submitted, just as a compiled dylib, not as source code.


## 1. Public Commit Record

| Date (UTC)   | Repo                           | Commit                  | Significance                                                                                   |
|--------------|--------------------------------|-------------------------|-----------------------------------------------------------------------------------------------|
| Oct 9        | `bitanath/rsd-swissgov`        | `9930ff9`               | My first public commit                                                                         |
| Oct 11       | `bitanath/rsd-swissgov`        | `4c7103a`               | Public `ABLATION.md`: 32-layer ablation, peak L15, band 13–17, two-prefill design, Zig engine  |
| Oct 13 10:29 | `amunozo/apertus-swissgov-rsd` | `8767408`               | Winner's template-only initial commit                                                         |
| Oct 13 10:47 | `amunozo/apertus-swissgov-rsd` | `c7736ca`               | Winner's first solution code                                                                   |
| Oct 13 13:49 | `amunozo/apertus-swissgov-rsd` | `c4d6b0d`               | Winner's dev DiffAlign layer extraction/surprisal                                             |
| Oct 14       | `amunozo/apertus-swissgov-rsd` | `57ab49b` / `final-run` | Winner's test predictions/test run                                                             |


My public history: **97 commits** from Oct 9. The winner's public history: **19 commits**, starting with a template-only commit and moving to solution code 18 minutes later.

## 2. Order of Work

1. **Public priority is established.** My layer ablation and two-prefill design were public Oct 11. The winner's first solution commit is Oct 13 10:47. That is the fact the winner must explain.
2. **The overlap is specific.** The winner uses frozen Apertus middle-layer DiffAlign in the same band I published, plus a supervised pair probe and CPU emphasis. The winner's report cites the paper and other baselines — not my public ablation. Ho
3. **The winner's public history does not show development.** Template-only initial commit, then solution code 18 minutes later, then the layer sweep the same day. The public record shows upload, not development. Provenance evidence is required.

## 3. Metric Record

| Comparison                             | Me        | Winner   | Gap             | Note                                                                              |
|----------------------------------------|-----------|----------|-----------------|-----------------------------------------------------------------------------------|
| Test (56/pair)                         | **39.5**  | **37.4** | **+2.1 me**     | Test favors me; winner's CI [31.9, 42.4] includes 39.5                            |
| Best-case dev                          | **46.93** | **39.9** | **+7.03 me**    | My best dev vs the winner's best dev (fudged dataset/in-sample testing)           |
| Winner's best dev vs my validated test | **39.5**  | **39.9** | **+0.4 winner** | The winner's most optimistic number barely clears my **actual unbiased test set** |

### For further detail here are all the published metrics as of date of results

| Number    | Split            | Protocol                                    |
|-----------|------------------|---------------------------------------------|
| **46.93** | dev/val, 34/pair | my best-case dev run                        |
| **39.9**  | dev, 168/pair    | winner's 5-fold OOF, config selected on dev  |
| **38.0**  | dev/val, 34/pair | winner's fit on dev/train, eval dev/val      |
| **39.5**  | test, 56/pair    | my validated test                           |
| **37.4**  | test, 56/pair    | winner's single frozen test run             |

## 5. Bottom Line

My ablation was public **Oct 11**. The winner's first solution commit was **Oct 13 10:47**. The winner's public history starts with a template-only commit and the solution appears 18 minutes later.

**Purely On metrics:**

- Test set macro Spearman **39.5 vs 37.4** → **+2.1 me**.
- Best-case dev/val fudgy dataset **46.93 vs 39.9** → **+7.03 me**.

I am just asking for an honest conversation guys. You don't need to play hardball with me or whatever. Just tell me your reasoning etc and we can talk about this.
