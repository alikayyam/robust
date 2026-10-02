"""
Unclipped re-run of the toy robust/non-robust-feature experiments, for
feature_reliance_paper.tex. Same task parameters as the companion root-cause
paper: x_i = (y - 1/2) * dmu_i + sigma_i * xi, with the reliable dimension
(dmu=1.0, sigma=sigma0=0.3) and five weak dimensions (dmu=alpha=0.4, sigma=0.25).
No clipping, so the Bayes rule is exactly linear. Models, optimiser and FGSM
follow the earlier note (8 ReLU units, Adam lr 0.02, 500 steps, wd 1e-2).

Parts: main (A/B/C, FGSM by attacked subset), bayes (accuracy + permutation
importance), reg (matched-accuracy regulariser comparison), phase (grid),
modelc (dead-weight scale). Results -> feature_reliance_unclipped.pkl.
"""

import copy
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

N_TRAIN, N_TEST = 4000, 1000
SIGMA0, ALPHA, SIGMA, WD = 0.3, 0.4, 0.25, 1e-2
EPS_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
SEEDS = list(range(10))
MASKS = {
    "all": [1, 1, 1, 1, 1, 1],
    "reliable": [1, 0, 0, 0, 0, 0],
    "weak5": [0, 1, 1, 1, 1, 1],
    "weak1": [0, 1, 0, 0, 0, 0],
}


def make(n, sigma0=SIGMA0, alpha=ALPHA, gen=None):
    y = torch.randint(0, 2, (n,), generator=gen)
    s = (y.float() - 0.5).unsqueeze(1)
    x0 = s * 1.0 + sigma0 * torch.randn(n, 1, generator=gen)
    xw = s * alpha + SIGMA * torch.randn(n, 5, generator=gen)
    return torch.cat([x0, xw], 1), y


def bayes_w(sigma0=SIGMA0, alpha=ALPHA):
    return torch.tensor([1.0 / sigma0 ** 2] + [alpha / SIGMA ** 2] * 5)


def bayes_acc(x, y, dims):
    w = bayes_w()
    llr = (x[:, dims] * w[dims]).sum(1)
    return ((llr > 0).long() == y).float().mean().item()


class MLP(nn.Module):
    def __init__(self, d, h=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, x):
        return self.net(x)


def train(m, x, y, wd=WD, l1=0.0, gl=0.0, steps=500, lr=0.02):
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = ce(m(x), y)
        if l1:
            loss = loss + l1 * m.net[0].weight.abs().sum()
        if gl:
            loss = loss + gl * m.net[0].weight.norm(dim=0).sum()
        loss.backward()
        opt.step()
    return m


def acc(m, x, y):
    with torch.no_grad():
        return (m(x).argmax(1) == y).float().mean().item()


def fgsm(m, x, y, eps, mask):
    with torch.no_grad():
        ok = m(x).argmax(1) == y
    xc, yc = x[ok], y[ok]
    xa = xc.clone().requires_grad_(True)
    nn.functional.cross_entropy(m(xa), yc).backward()
    adv = xc + eps * xa.grad.sign() * torch.tensor(mask, dtype=torch.float32)
    with torch.no_grad():
        return (m(adv).argmax(1) != yc).float().mean().item()


def ms(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def wnorm_ratio(m):
    n = m.net[0].weight.data.norm(dim=0).tolist()
    return n[0] / (sum(n[1:]) / 5 + 1e-8)


def data(seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    return make(N_TRAIN, gen=g), make(N_TEST, gen=g)


def part_main():
    out = {k: [] for k in ["accA", "accB", "accC", "bayes"]}
    for k in MASKS:
        for e in EPS_GRID:
            out[("A", k, e)] = []
    for e in EPS_GRID:
        out[("B", "reliable", e)] = []
        out[("C", "all", e)] = []
        out[("C", "dead", e)] = []
    for s in SEEDS:
        (xt, yt), (xs, ys) = data(s)
        torch.manual_seed(s)
        A = train(MLP(6), xt, yt)
        torch.manual_seed(s)
        B = train(MLP(1), xt[:, :1], yt)
        xtc, xsc = xt.clone(), xs.clone()
        xtc[:, 1:] = 0
        xsc[:, 1:] = 0
        torch.manual_seed(s)
        C = train(MLP(6), xtc, yt)
        out["accA"].append(acc(A, xs, ys)); out["accB"].append(acc(B, xs[:, :1], ys))
        out["accC"].append(acc(C, xsc, ys)); out["bayes"].append(bayes_acc(xs, ys, list(range(6))))
        for e in EPS_GRID:
            for k, mk in MASKS.items():
                out[("A", k, e)].append(fgsm(A, xs, ys, e, mk))
            out[("B", "reliable", e)].append(fgsm(B, xs[:, :1], ys, e, [1]))
            out[("C", "all", e)].append(fgsm(C, xsc, ys, e, MASKS["all"]))
            out[("C", "dead", e)].append(fgsm(C, xsc, ys, e, MASKS["weak5"]))
        print("main seed", s, flush=True)
    return out


def part_bayes():
    d = {"bayes_all": [], "bayes_rel": [], "bayes_w5": [], "bayes_w1": [], "mlp": [],
         "pb_rel": [], "pb_w": [], "pm_rel": [], "pm_w": []}
    for s in SEEDS:
        (xt, yt), (xs, ys) = data(s)
        torch.manual_seed(s)
        A = train(MLP(6), xt, yt)
        d["bayes_all"].append(bayes_acc(xs, ys, list(range(6))))
        d["bayes_rel"].append(bayes_acc(xs, ys, [0]))
        d["bayes_w5"].append(bayes_acc(xs, ys, [1, 2, 3, 4, 5]))
        d["bayes_w1"].append(bayes_acc(xs, ys, [1]))
        d["mlp"].append(acc(A, xs, ys))
        g = torch.Generator().manual_seed(500 + s)
        base_b, base_m = d["bayes_all"][-1], d["mlp"][-1]

        def perm(cols):
            x = xs.clone()
            for c in cols:
                x[:, c] = x[torch.randperm(len(x), generator=g), c]
            return x
        xr, xw = perm([0]), perm([1, 2, 3, 4, 5])
        d["pb_rel"].append(base_b - bayes_acc(xr, ys, list(range(6))))
        d["pb_w"].append(base_b - bayes_acc(xw, ys, list(range(6))))
        d["pm_rel"].append(base_m - acc(A, xr, ys))
        d["pm_w"].append(base_m - acc(A, xw, ys))
        print("bayes seed", s, flush=True)
    return d


def sweep(name, strengths, kw):
    res = []
    for st in strengths:
        r = dict(strength=st, acc=[], ratio=[], rel=[], weak=[])
        for s in SEEDS:
            (xt, yt), (xs, ys) = data(s)
            torch.manual_seed(s)
            m = train(MLP(6), xt, yt, **kw(st))
            r["acc"].append(acc(m, xs, ys)); r["ratio"].append(wnorm_ratio(m))
            r["rel"].append(fgsm(m, xs, ys, 0.3, MASKS["reliable"]))
            r["weak"].append(fgsm(m, xs, ys, 0.3, MASKS["weak5"]))
        res.append(r)
        print(name, st, ms(r["acc"])[0], ms(r["weak"])[0], flush=True)
    return res


def part_reg():
    out = {}
    out["L1"] = sweep("L1", [0.02, 0.05, 0.1, 0.2, 0.4, 0.8], lambda s: dict(wd=0.0, l1=s))
    out["L2"] = sweep("L2", [1e-2, 5e-2, 1e-1, 2e-1, 4e-1, 8e-1], lambda s: dict(wd=s))
    out["GL"] = sweep("GL", [0.01, 0.02, 0.05, 0.1, 0.2, 0.4], lambda s: dict(wd=0.0, gl=s))
    out["EN"] = sweep("EN", [0.0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1], lambda s: dict(wd=s, l1=0.1))
    prune = []
    for k in range(6):
        r = dict(strength=k, acc=[], ratio=[], rel=[], weak=[])
        for s in SEEDS:
            (xt, yt), (xs, ys) = data(s)
            torch.manual_seed(s)
            m = train(MLP(6), xt, yt)
            order = sorted(range(1, 6), key=lambda i: m.net[0].weight.data[:, i].norm().item())
            with torch.no_grad():
                for i in order[:k]:
                    m.net[0].weight.data[:, i] = 0.0
            r["acc"].append(acc(m, xs, ys)); r["ratio"].append(wnorm_ratio(m))
            r["rel"].append(fgsm(m, xs, ys, 0.3, MASKS["reliable"]))
            r["weak"].append(fgsm(m, xs, ys, 0.3, MASKS["weak5"]))
        prune.append(r)
        print("prune", k, ms(r["acc"])[0], ms(r["weak"])[0], flush=True)
    out["prune"] = prune
    return out


def part_phase():
    out = {}
    for s0 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for a in [0.1, 0.2, 0.4, 0.6]:
            for wd in [0.0, 1e-3, 1e-2, 1e-1]:
                r = dict(acc=[], ratio=[], rel=[], weak=[])
                for s in range(5):
                    g = torch.Generator().manual_seed(s)
                    torch.manual_seed(s)
                    xt, yt = make(N_TRAIN, s0, a, g)
                    xs, ys = make(N_TEST, s0, a, g)
                    m = train(MLP(6), xt, yt, wd=wd)
                    r["acc"].append(acc(m, xs, ys)); r["ratio"].append(wnorm_ratio(m))
                    r["rel"].append(fgsm(m, xs, ys, 0.2, MASKS["reliable"]))
                    r["weak"].append(fgsm(m, xs, ys, 0.2, MASKS["weak5"]))
                out[(s0, a, wd)] = r
                print("phase", s0, a, wd, ms(r["weak"])[0], flush=True)
    return out


def part_modelc():
    out = {}
    for scale in [1, 3, 10, 100]:
        v = []
        for s in SEEDS:
            (xt, yt), (xs, ys) = data(s)
            xtc, xsc = xt.clone(), xs.clone()
            xtc[:, 1:] = 0
            xsc[:, 1:] = 0
            torch.manual_seed(s)
            m = MLP(6)
            with torch.no_grad():
                m.net[0].weight.data[:, 1:] *= scale
            train(m, xtc, yt)
            v.append(fgsm(m, xsc, ys, 0.3, MASKS["weak5"]))
        out[scale] = v
        print("modelc", scale, ms(v), flush=True)
    return out


if __name__ == "__main__":
    parts = sys.argv[1:] or ["main", "bayes", "reg", "phase", "modelc"]
    res = {}
    for p in parts:
        res[p] = globals()["part_" + p]()
        pickle.dump(res, open(f"feature_reliance_unclipped_{'_'.join(parts)}.pkl", "wb"))
