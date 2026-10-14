#!/usr/bin/env python3
"""Joint pair probe on frozen Apertus features: layer fusion + seed ensembling.

Merged from the earlier single-seed probe (which is what saved weights) and the
layer-fusion/seed-ensemble variant. Saves the best-val-epoch weights by default
to layerwise/pairprobe_best_L<layers>_<fusion>.pth.

Usage: transform.py <layers e.g. 15 | 14,15,16,17> <sum|concat> <seeds> [epochs] [--objective bce|focal|pair|bce+pair]
Reports per-language val rho (pooled both sides) + synthetic OOD rho.
"""
import sys, os, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm

OUT = Path(os.environ.get("LW", "./aperdiff/python/layerwise"))
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CEIL = {"de": 0.561, "fr": 0.436, "it": 0.651}
DIM = 4096


def load_pairs(role, layers, fusion):
    pairs, skipped = [], 0
    first = layers[0]
    keep = [s for s in os.environ.get("KEEP", "").split(",") if s]   # NOTE: KEEP=de,fr limits which languages load
    for p in sorted(OUT.glob(f"{role}_*_L{first}.pt")):
        if p.name.endswith("_self_L15.pt"):
            continue
        stem = p.name[len(role) + 1:-(len(f"_L{first}.pt"))]
        if keep and stem not in keep:
            continue
        per = [torch.load(OUT / f"{role}_{stem}_L{l}.pt", map_location="cpu", weights_only=False) for l in layers]
        for i in range(len(per[0])):
            e0 = per[0][i]
            fa = [per[j][i]["feats_a"] for j in range(len(layers))]
            fb = [per[j][i]["feats_b"] for j in range(len(layers))]
            na, nb = fa[0].shape[0], fb[0].shape[0]
            if na + nb == 0:
                skipped += 1; continue   # NOTE: empty pair -> zero-length seq crashes the GRU
            if na + nb > 2500:
                skipped += 1; continue
            if fusion == "sum":
                A = sum(x.float() for x in fa) / len(layers); B = sum(x.float() for x in fb) / len(layers)
            else:
                A = torch.cat([x.float() for x in fa], -1); B = torch.cat([x.float() for x in fb], -1)
            ya = np.asarray(e0["labels_a"], np.float32)[:na]; yb = np.asarray(e0["labels_b"], np.float32)[:nb]
            pairs.append((stem, A.half(), B.half(), ya, yb))
        del per
    if skipped:
        print(f"  [{role}] skipped {skipped} pairs (empty or >2500 tok)", flush=True)
    return pairs


def collate(batch, d_in):
    L = max(1, max(a.shape[0] + b.shape[0] for _, a, b, _, _ in batch)); B = len(batch)
    X = torch.zeros(B, L, d_in, dtype=torch.float16); S = torch.zeros(B, L, dtype=torch.long)
    Y = torch.full((B, L), -1.0); P = torch.ones(B, L, dtype=torch.bool)
    for i, (_, a, b, ya, yb) in enumerate(batch):
        na, nb = a.shape[0], b.shape[0]
        X[i, :na] = a; X[i, na:na + nb] = b; S[i, na:na + nb] = 1
        Y[i, :na] = torch.tensor(ya); Y[i, na:na + nb] = torch.tensor(yb); P[i, :na + nb] = False
    return X, S, Y, P


def make_loss(objective, gamma=2.0, margin=1.0, lam=1.0, hi_thr=0.5, lo_thr=0.05):
    """Loss over a batch of documents. y uses -1 for padding/junk (excluded).

    bce      : token-pooled soft-target BCE (identical to the earlier baseline)
    focal    : gamma-modulated BCE (counters the rare-positive imbalance)
    pair     : within-doc pairwise margin; ranks hi-label tokens above lo-label ones
    bce+pair : BCE (cross-doc scale) + pairwise (within-doc order)
    """
    def bce_of(s, t):
        return nn.functional.binary_cross_entropy_with_logits(s, t)

    def fn(logits, y):
        valid = y >= 0
        if objective in ("bce", "focal"):
            s = logits[valid]; t = y[valid].clamp(0, 1)
            if len(s) == 0:
                return None
            if objective == "bce":
                return bce_of(s, t)
            p = torch.sigmoid(s); pt = p * t + (1 - p) * (1 - t)
            return (((1 - pt) ** gamma) * nn.functional.binary_cross_entropy_with_logits(s, t, reduction="none")).mean()

        losses = []
        any_pair = False
        for b in range(logits.shape[0]):
            yb = y[b]; vb = yb >= 0
            if vb.sum() == 0:
                continue
            sb = logits[b][vb]; tb = yb[vb].clamp(0, 1)
            hi = torch.nonzero(yb >= hi_thr, as_tuple=True)[0]
            lo = torch.nonzero((yb >= 0) & (yb <= lo_thr), as_tuple=True)[0]
            if len(hi) > 0 and len(lo) > 0:
                if len(hi) > 400: hi = hi[torch.randperm(len(hi))[:400]]
                if len(lo) > 400: lo = lo[torch.randperm(len(lo))[:400]]
                diff = logits[b][hi].unsqueeze(1) - logits[b][lo].unsqueeze(0)  # (H, Lo)
                pair = nn.functional.softplus(margin - diff).mean()
                any_pair = True
            else:
                pair = logits.sum() * 0.0  # grad-carrying no-op
            losses.append(pair if objective == "pair" else bce_of(sb, tb) + lam * pair)
        if objective == "pair" and not any_pair:
            return None
        return torch.stack(losses).mean() if losses else None

    return fn


class PairProbe(nn.Module):
    def __init__(self, d_in, d_model=512, nhead=8, nlayers=4, ff=2048, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.seg = nn.Embedding(2, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ff, dropout=dropout,
                                         batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x, seg, pad):
        h = self.proj(x) + self.seg(seg)
        h = self.enc(h, src_key_padding_mask=pad)
        return self.head(h).squeeze(-1)


class PairProbeMLP(nn.Module):
    """Cheapest baseline; tokens processed independently (no cross-side info)."""
    def __init__(self, d_in, d_model=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.seg = nn.Embedding(2, d_model)
        self.block = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(2 * d_model, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x, seg, pad):
        h = self.proj(x) + self.seg(seg)
        h = self.norm(h + self.block(h))
        return self.head(h).squeeze(-1).masked_fill(pad, 0.0)

## FIXME This is a backup, in case the OOD is too bad on GRU
class PairProbeCNN(nn.Module):
    """Dilated CNN: local patterns with growing receptive field."""
    def __init__(self, d_in, d_model=512, dropout=0.1, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.seg = nn.Embedding(2, d_model)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=3, padding=d, dilation=d),
                          nn.GELU(), nn.Dropout(dropout), nn.Conv1d(d_model, d_model, kernel_size=1))
            for d in dilations])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in dilations])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x, seg, pad):
        h = self.proj(x) + self.seg(seg)
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        for block, norm in zip(self.blocks, self.norms):
            z = block(h.transpose(1, 2)).transpose(1, 2)
            h = norm(h + z)
            h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        return self.head(h).squeeze(-1).masked_fill(pad, 0.0)


## TODO use this, it appears to have the best decoding capability 
class PairProbeGRU(nn.Module):
    """Bidirectional GRU: sequential context, no self-attention."""
    def __init__(self, d_in, d_model=512, dropout=0.1):
        super().__init__()
        assert d_model % 2 == 0
        self.proj = nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.seg = nn.Embedding(2, d_model)
        self.rnn = nn.GRU(d_model, d_model // 2, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x, seg, pad):
        h = self.proj(x) + self.seg(seg)
        h, _ = self.rnn(h)
        h = self.norm(h)
        return self.head(h).squeeze(-1).masked_fill(pad, 0.0)


class PairProbeXAttn(nn.Module):
    """Alternating self-attention + cross-attention (A tokens attend only to B, and vice-versa)."""
    def __init__(self, d_in, d_model=512, nhead=8, nlayers=2, ff=2048, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.seg = nn.Embedding(2, d_model)
        self.nhead = nhead
        self.slf = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                                             batch_first=True, norm_first=True, activation="gelu")
                                  for _ in range(nlayers)])
        self.xa = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                                 for _ in range(nlayers)])
        self.xn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(nlayers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x, seg, pad):
        h = self.proj(x) + self.seg(seg)
        same = seg.unsqueeze(1) == seg.unsqueeze(2)                 # (B,L,L) True => disallow in cross attn
        am = same.repeat_interleave(self.nhead, 0)                  # (B*nhead,L,L)
        for slf, xa, xn in zip(self.slf, self.xa, self.xn):
            h = slf(h, src_key_padding_mask=pad)
            z, _ = xa(h, h, h, attn_mask=am, key_padding_mask=pad, need_weights=False)
            h = xn(h + z)
        return self.head(h).squeeze(-1).masked_fill(pad, 0.0)


class PairProbeBiAlign(nn.Module):
    """Explicit cross-side alignment: bilinear sim(A_i, B_j) -> row-max + soft attention."""
    def __init__(self, d_in, d_model=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.seg = nn.Embedding(2, d_model)
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5
        self.mix = nn.Sequential(nn.Linear(2 * d_model + 1, d_model), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x, seg, pad):
        h = self.proj(x) + self.seg(seg)
        q = self.W(h)                                               # (B,L,d)
        sim = (q @ h.transpose(1, 2)) * self.scale                  # (B,L,L)
        opp = (seg.unsqueeze(1) != seg.unsqueeze(2)) & (~pad.unsqueeze(1))   # allowed keys
        sim_m = sim.masked_fill(~opp, torch.finfo(sim.dtype).min)
        maxv = sim_m.max(dim=2).values
        maxv = torch.where(opp.any(dim=2), maxv, torch.zeros_like(maxv))
        attn = torch.softmax(sim_m, dim=2)
        attnv = attn @ h                                            # (B,L,d)
        feat = self.mix(torch.cat([h, attnv, maxv.unsqueeze(-1)], dim=-1))
        return self.head(feat).squeeze(-1).masked_fill(pad, 0.0)


ARCHS = {"transformer": PairProbe, "mlp": PairProbeMLP, "cnn": PairProbeCNN,
         "gru": PairProbeGRU, "xattn": PairProbeXAttn, "bilinear": PairProbeBiAlign}


def build_model(arch, d_in):
    return ARCHS[arch](d_in)


@torch.no_grad()
def predict(model, pairs, d_in, bs=2):
    model.eval(); groups = {}
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i + bs]
        X, S, Y, P = collate(batch, d_in)
        out = torch.sigmoid(model(X.float().to(DEV), S.to(DEV), P.to(DEV))).cpu()
        for j, (lang, *_) in enumerate(batch):
            m = (~P[j]) & (Y[j] >= 0)
            groups.setdefault(lang, ([], []))
            groups[lang][0].append(out[j][m].numpy()); groups[lang][1].append(Y[j][m].numpy())
    return {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in groups.items()}


def run_arch(arch, tr, va, sy, d_in, layers, fusion, objective, seeds, epochs,
             loss_fn, dead_after, dead_below, roles=("train", "val", "syn"), save_tag=None, te=None):
    Vp, Op, Tp = defaultdict(list), defaultdict(list), defaultdict(list); vl = {}; ol = {}; tl = {}
    seed_bests, gbest, aborted = [], (-9.0, None, -1, -1), False
    for s in range(seeds):
        torch.manual_seed(s)
        model = build_model(arch, d_in).to(DEV)
        npar = sum(p.numel() for p in model.parameters())
        if s == 0:
            print(f"  [{arch}] params={npar/1e6:.2f}M", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        # NOTE: optional ReduceLROnPlateau on the held-out val macro (env LR_PATIENCE>0).
        lr_pat = int(os.environ.get("LR_PATIENCE", "0"))
        sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="max", factor=float(os.environ.get("LR_FACTOR", "0.1")), patience=lr_pat)
                 if lr_pat > 0 else None)
        best, bep, best_pv, best_state = -9, -1, None, None
        for ep in range(epochs):
            model.train(); perm = torch.randperm(len(tr)).tolist(); tot = 0.0
            bar = tqdm(range(0, len(perm), 2), desc=f"{arch} s{s} ep{ep:02d}", ncols=100, leave=False)
            for i in bar:
                X, Sg, Y, P = collate([tr[k] for k in perm[i:i + 2]], d_in)
                logits = model(X.float().to(DEV), Sg.to(DEV), P.to(DEV))
                loss = loss_fn(logits, Y.to(DEV))
                if loss is None:
                    continue
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tot += float(loss.detach())
                bar.set_postfix(loss=f"{tot/(i//2+1):.4f}")
            if DEV == "mps": torch.mps.empty_cache()
            pv = predict(model, va, d_in)
            vkeys = sorted(pv.keys())
            macro = np.mean([spearmanr(*pv[k]).statistic for k in vkeys])
            print(f"  [{arch}] s{s} ep{ep:02d} loss={tot/max(1,len(perm)//2):.4f} "
                  + " ".join(f"{k}={spearmanr(*pv[k]).statistic:.3f}" for k in vkeys)
                  + f" macro={macro:.4f}", flush=True)
            if sched is not None:
                sched.step(macro)
                print(f"  [{arch}] s{s} ep{ep:02d} lr={opt.param_groups[0]['lr']:.2e}", flush=True)
            if macro > best:
                best, bep, best_pv = macro, ep, pv
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) >= dead_after and best < dead_below:
                print(f"  [{arch}] DEAD after {ep+1} eps (best macro={best:.4f} < {dead_below}) -> abort", flush=True)
                aborted = True; break
        if aborted:
            del model
            if DEV == "mps": torch.mps.empty_cache()
            break
        print(f"  [{arch}] seed{s}: best val macro={best:.4f} @ep{bep}", flush=True)
        seed_bests.append(best)
        if best > gbest[0]:
            gbest = (best, best_state, s, bep)
        for g, (p, l) in best_pv.items(): Vp[g].append(p); vl[g] = l
        model.load_state_dict(best_state)   # OOD must use the best epoch, not the final one
        od = predict(model, sy, d_in)
        print(f"  [{arch}] seed{s}: OOD " + " ".join(f"{g}={spearmanr(*od[g]).statistic:.3f}" for g in sorted(od)), flush=True)
        for g, (p, l) in od.items(): Op[g].append(p); ol[g] = l
        if te is not None:   # NOTE: optional held-out TEST role, scored per seed
            td = predict(model, te, d_in)
            print(f"  [{arch}] seed{s}: TEST " + " ".join(f"{g}={spearmanr(*td[g]).statistic:.4f}" for g in sorted(td)), flush=True)
            for g, (p, l) in td.items(): Tp[g].append(p); tl[g] = l
        del model
        if DEV == "mps": torch.mps.empty_cache()

    res = {"arch": arch, "aborted": aborted, "seed_bests": seed_bests}
    if aborted or gbest[1] is None:
        return res
    res["val"] = {g: spearmanr(np.mean(Vp[g], 0), vl[g]).statistic for g in sorted(Vp.keys())}
    res["val_macro"] = float(np.mean(list(res["val"].values())))
    res["ood"] = {g: spearmanr(np.mean(Op[g], 0), ol[g]).statistic for g in sorted(Op.keys())}
    res["ood_macro"] = float(np.mean(list(res["ood"].values())))
    if te is not None:
        res["test"] = {g: spearmanr(np.mean(Tp[g], 0), tl[g]).statistic for g in sorted(Tp.keys())}
        res["test_macro"] = float(np.mean(list(res["test"].values())))
    tag = f"L{'-'.join(map(str, layers))}_{fusion}_{objective}_{arch}"
    if save_tag:
        tag += f"_{save_tag}"
    elif list(roles) != ["train", "val", "syn"]:
        tag += "_" + "-".join(roles)
    mp = OUT / f"pairprobe_best_{tag}.pth"
    torch.save({"state_dict": gbest[1], "val_macro": gbest[0], "arch": arch, "seed": gbest[2],
                "epoch": gbest[3], "layers": layers, "fusion": fusion, "objective": objective, "d_in": d_in}, mp)
    res["path"] = str(mp)
    print(f"  [{arch}] saved -> {mp.name}", flush=True)
    return res


def main():
    layers = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "15").split(",")]
    fusion = sys.argv[2] if len(sys.argv) > 2 else "sum"
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    objective, archs, dead_after, dead_below = "bce", "transformer", 5, 0.33
    val_frac, save_tag = 0.0, None
    for j, a in enumerate(sys.argv):
        nxt = sys.argv[j + 1] if j + 1 < len(sys.argv) else None
        if a.startswith("--objective="): objective = a.split("=", 1)[1]
        elif a == "--objective" and nxt: objective = nxt
        elif a.startswith("--arch="): archs = a.split("=", 1)[1]
        elif a == "--arch" and nxt: archs = nxt
        elif a == "--dead-after" and nxt: dead_after = int(nxt)
        elif a == "--dead-below" and nxt: dead_below = float(nxt)
        elif a == "--val-frac" and nxt: val_frac = float(nxt)
        elif a.startswith("--val-frac="): val_frac = float(a.split("=", 1)[1])
        elif a == "--tag" and nxt: save_tag = nxt
        elif a.startswith("--tag="): save_tag = a.split("=", 1)[1]
    assert objective in ("bce", "focal", "pair", "bce+pair"), f"bad objective {objective}"
    loss_fn = make_loss(objective)
    d_in = DIM * len(layers) if fusion == "concat" else DIM
    arches = [a for a in archs.split(",") if a]
    for a in arches:
        assert a in ARCHS, f"bad arch {a} (choose from {list(ARCHS)})"
    print(f"=== layers={layers} fusion={fusion} seeds={seeds} epochs={epochs} "
          f"objective={objective} archs={arches} d_in={d_in} ===", flush=True)
    roles = (os.environ.get("ROLES", "train,val,syn")).split(",")
    tr = load_pairs(roles[0], layers, fusion)
    if val_frac > 0:
        # NOTE: carve an in-domain held-out slice from the training role so early
        # stopping never peeks at the reporting/transfer set (roles[1] is ignored).
        gen = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(tr), generator=gen).tolist()
        nv = max(1, int(round(len(tr) * val_frac)))
        va = [tr[i] for i in perm[:nv]]; tr = [tr[i] for i in perm[nv:]]
        print(f"  [val-frac={val_frac}] held out {nv} pairs of '{roles[0]}' for early stopping", flush=True)
    else:
        va = load_pairs(roles[1], layers, fusion)
    sy = load_pairs(roles[2], layers, fusion)
    test_role = os.environ.get("TEST_ROLE", "")
    te = load_pairs(test_role, layers, fusion) if test_role else None
    print(f"roles {roles} | pairs {len(tr)}/{len(va)}/{len(sy)}"
          + (f" | test({test_role})={len(te)}" if te is not None else ""), flush=True)

    results = []
    for arch in arches:
        print(f"\n########## ARCH: {arch} ##########", flush=True)
        results.append(run_arch(arch, tr, va, sy, d_in, layers, fusion, objective,
                                seeds, epochs, loss_fn, dead_after, dead_below, roles, save_tag, te))

    print("\n================= SUMMARY =================", flush=True)
    for r in results:
        if r.get("aborted") or "val_macro" not in r:
            print(f"{r['arch']:<12}ABORTED", flush=True); continue
        # NOTE: print whatever language keys exist (KEEP=de yields only 'de', etc.)
        v = "  ".join(f"{k}={r['val'][k]:.4f}" for k in sorted(r["val"]))
        o = "  ".join(f"{k}={r['ood'][k]:.4f}" for k in sorted(r.get("ood", {})))
        line = f"{r['arch']:<12} val_macro={r['val_macro']:.4f}  [{v}]  | OOD_macro={r['ood_macro']:.4f}  [{o}]"
        if "test_macro" in r:
            t = "  ".join(f"{k}={r['test'][k]:.4f}" for k in sorted(r["test"]))
            line += f"  | TEST_macro={r['test_macro']:.4f}  [{t}]"
        print(line, flush=True)


if __name__ == "__main__":
    main()
