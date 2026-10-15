# Apertus layer bands

A general mental model of how a decoder spreads work (surface → syntax →
semantics → task → output). The names are heuristic; the
[32-layer ablation](../technical_report.md#61-intermediate-layer-ablation-vs-human-agreement)
is what turns them into evidence — the signal peaks in the *Semantic build-up*
band (layers 14–17).

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

Apertus is a dense transformer, not an MoE: every block encodes information about
the input tokens. We distill the intermediate representations of its multilingual
understanding (the L14–17 band) to give the pair-probe a strong, stable signal.
