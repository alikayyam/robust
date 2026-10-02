"""
Minimal experiments for the "weak-feature accumulation" account of
adversarial vulnerability (companion to root_cause_paper.tex).

Setup. Inputs are unclipped Gaussians, conditionally independent given a
balanced binary label y:

    x_i = (y - 1/2) * dmu_i + sigma_i * eps_i,      eps_i ~ N(0, 1)

so the Bayes-optimal classifier is *exactly linear*,
    LLR(x) = sum_i w_i x_i,   w_i = dmu_i / sigma_i^2,
and the exact minimal L_p perturbation that flips it is
    min-eps_inf = |LLR| / ||w||_1,    min-eps_2 = |LLR| / ||w||_2.
Dropping the [0,1] clipping of the earlier paper is deliberate: it makes the
Bayes rule exactly linear and removes the "eps=0.5 spans the whole domain"
artifact, because the domain is unbounded.

Vulnerability metric. Instead of attack success at a grid of eps, we compute
for every correctly-classified test point the minimal eps that flips it
(exact for Bayes; per-point bisection over PGD for trained networks). The
ECDF of this quantity *is* the attack-success-vs-eps curve.

Experiments (see the paper):
  E1  same information spread over k dims   (Bayes exact + trained MLP)
  E2  attack the Bayes classifier itself    (is vulnerability intrinsic?)
  E3  who is vulnerable: attacker restricted to a subset of dims
  E4  interventions: remove / shrink the weak dims, weak-only model, L1 sweep
Results are written to root_cause_results.pkl; figures by root_cause_figures.py.
"""

import math
import pickle
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

N_TRAIN, N_TEST = 4000, 2000
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))
HIDDEN, LR, STEPS, WD = 8, 0.02, 500, 1e-2

# E2-E4 configuration: same generative parameters as the earlier paper
SIG = dict(dmu=1.0, sigma=0.3)          # strong dim, d' = 3.33
WEAK = dict(dmu=0.4, sigma=0.25)        # weak dims, d' = 1.6 each
N_WEAK = 5

# E1 configuration: fixed total information, spread over k dims
D_TOTAL = 1.0 / 0.3                     # same total d' as the strong dim above
E1_SIGMA = 0.25
E1_KS = [1, 2, 4, 8, 16, 32, 64]


# ---------------------------------------------------------------- data / Bayes
def make_data(n, dmu, sigma, gen):
    """dmu, sigma: tensors (D,). Returns x (n,D), y (n,)."""
    y = torch.randint(0, 2, (n,), generator=gen)
    eps = torch.randn(n, len(dmu), generator=gen)
    x = (y.float().unsqueeze(1) - 0.5) * dmu + sigma * eps
    return x, y


def bayes_weights(dmu, sigma):
    return dmu / sigma ** 2


def bayes_accuracy_closed_form(dmu, sigma):
    dprime = torch.sqrt(((dmu / sigma) ** 2).sum()).item()
    return 0.5 * (1 + math.erf(dprime / 2 / math.sqrt(2)))


def bayes_min_eps(x, y, w, norm, subset=None):
    """Exact minimal flip radius of the linear Bayes rule, correctly
    classified points only. subset: boolean mask (D,) of attackable dims."""
    llr = x @ w
    pred = (llr > 0).long()
    ok = pred == y
    wv = w if subset is None else w * subset
    denom = wv.abs().sum() if norm == "inf" else wv.norm()
    if denom == 0:
        return torch.full((int(ok.sum()),), float("inf"))
    return llr[ok].abs() / denom


class LinearBayes(nn.Module):
    """The Bayes rule as a torch module so the generic attack can be run on
    it (used only to validate the closed-form min-eps against PGD)."""

    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, x):
        l = x @ self.w
        return torch.stack([-l / 2, l / 2], dim=1)


# ---------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 2))

    def forward(self, x):
        return self.net(x)


def train_mlp(x, y, l1=0.0, wd=WD):
    m = MLP(x.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    for _ in range(STEPS):
        opt.zero_grad()
        loss = ce(m(x), y)
        if l1 > 0:
            loss = loss + l1 * m.net[0].weight.abs().sum()
        loss.backward()
        opt.step()
    return m


def accuracy(m, x, y):
    with torch.no_grad():
        return (m(x).argmax(1) == y).float().mean().item()


# ---------------------------------------------------------------- attacks
def _project(delta, eps, norm):
    if norm == "inf":
        return torch.max(torch.min(delta, eps), -eps)
    n = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return delta * torch.clamp(eps / n, max=1.0)


def flips_at(model, x, y, eps, norm, mask, steps=20):
    """PGD with per-sample budget eps (n,1). True if the prediction is wrong
    at ANY iterate (best-of-trajectory). mask: (D,) 0/1 attackable dims."""
    n = len(x)
    delta = torch.zeros_like(x)
    flipped = torch.zeros(n, dtype=torch.bool)
    step = eps / 4.0
    for _ in range(steps):
        d = delta.clone().requires_grad_(True)
        logits = model(x + d)
        flipped |= logits.argmax(1) != y
        loss = nn.functional.cross_entropy(logits, y, reduction="sum")
        g = torch.autograd.grad(loss, d)[0] * mask
        with torch.no_grad():
            if norm == "inf":
                delta = delta + step * g.sign()
            else:
                gn = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                delta = delta + step * g / gn
            delta = _project(delta, eps, norm) * mask
    with torch.no_grad():
        flipped |= model(x + delta).argmax(1) != y
    return flipped


def min_eps(model, x, y, norm, mask=None, hi=4.0, iters=14):
    """Per-point minimal flip radius by bisection over PGD (correctly
    classified points only). Returns inf where even eps=hi cannot flip."""
    model.eval()
    if mask is None:
        mask = torch.ones(x.shape[1])
    with torch.no_grad():
        ok = model(x).argmax(1) == y
    x, y = x[ok], y[ok]
    n = len(x)
    lo = torch.zeros(n, 1)
    up = torch.full((n, 1), hi)
    reachable = flips_at(model, x, y, up, norm, mask)
    for _ in range(iters):
        mid = (lo + up) / 2
        f = flips_at(model, x, y, mid, norm, mask)
        up = torch.where(f.unsqueeze(1), mid, up)
        lo = torch.where(f.unsqueeze(1), lo, mid)
    out = up.squeeze(1)
    out[~reachable] = float("inf")
    return out


def first_layer_col_norms(m):
    return m.net[0].weight.data.norm(dim=0)


def summ(a):
    a = np.asarray(a)
    return float(np.median(a))


# ---------------------------------------------------------------- experiments
def seed_gen(seed, tag):
    g = torch.Generator()
    g.manual_seed(1000 * seed + tag)
    return g


def e1_seed(seed):
    """Same total information spread over k equally-informative dims."""
    out = {}
    for k in E1_KS:
        dprime_k = D_TOTAL / math.sqrt(k)
        dmu = torch.full((k,), dprime_k * E1_SIGMA)
        sigma = torch.full((k,), E1_SIGMA)
        g = seed_gen(seed, k)
        xtr, ytr = make_data(N_TRAIN, dmu, sigma, g)
        xte, yte = make_data(N_TEST, dmu, sigma, g)
        w = bayes_weights(dmu, sigma)
        torch.manual_seed(seed)
        m = train_mlp(xtr, ytr)
        r = dict(
            bayes_acc=((xte @ w > 0).long() == yte).float().mean().item(),
            mlp_acc=accuracy(m, xte, yte),
            w1=w.abs().sum().item(), w2=w.norm().item(),
            bayes_inf=summ(bayes_min_eps(xte, yte, w, "inf")),
            bayes_2=summ(bayes_min_eps(xte, yte, w, "2")),
            mlp_inf=summ(min_eps(m, xte, yte, "inf")),
            mlp_2=summ(min_eps(m, xte, yte, "2")),
        )
        out[k] = r
    return out


def e2_e3_e4_seed(seed):
    d = 1 + N_WEAK
    dmu = torch.tensor([SIG["dmu"]] + [WEAK["dmu"]] * N_WEAK)
    sigma = torch.tensor([SIG["sigma"]] + [WEAK["sigma"]] * N_WEAK)
    g = seed_gen(seed, 7)
    xtr, ytr = make_data(N_TRAIN, dmu, sigma, g)
    xte, yte = make_data(N_TEST, dmu, sigma, g)
    w = bayes_weights(dmu, sigma)
    res = dict(w=w.tolist())

    torch.manual_seed(seed)
    A = train_mlp(xtr, ytr)
    res["acc_A"] = accuracy(A, xte, yte)
    res["bayes_acc"] = ((xte @ w > 0).long() == yte).float().mean().item()
    res["colnorm_A"] = first_layer_col_norms(A).tolist()

    # validation of the closed form against PGD on the Bayes module
    lb = LinearBayes(w)
    ex = bayes_min_eps(xte, yte, w, "inf")
    pg = min_eps(lb, xte, yte, "inf")
    res["closed_form_vs_pgd_maxabs"] = float((ex - pg).abs().max())

    # ---- E2: min-eps arrays, Bayes vs trained A (kept for ECDFs)
    for norm in ("inf", "2"):
        res[f"bayes_{norm}"] = bayes_min_eps(xte, yte, w, norm).numpy()
        res[f"A_{norm}"] = min_eps(A, xte, yte, norm).numpy()

    # ---- E3: attacker restricted to a subset of dims
    subsets = {
        "signal (1 dim)": [0],
        "one weak dim": [1],
        "all 5 weak dims": [1, 2, 3, 4, 5],
        "all 6 dims": [0, 1, 2, 3, 4, 5],
    }
    res["E3"] = {}
    for name, idx in subsets.items():
        mask = torch.zeros(d)
        mask[idx] = 1
        res["E3"][name] = dict(
            bayes=bayes_min_eps(xte, yte, w, "inf", mask).numpy(),
            A=min_eps(A, xte, yte, "inf", mask).numpy(),
        )

    # ---- E4: interventions. B: signal only. W: weak only. A+L1 sweep.
    models = {}
    torch.manual_seed(seed)
    models["B: signal only"] = (train_mlp(xtr[:, :1], ytr), [0])
    torch.manual_seed(seed)
    models["W: weak only"] = (train_mlp(xtr[:, 1:], ytr), [1, 2, 3, 4, 5])
    res["E4"] = {}
    for name, (m, idx) in models.items():
        xs = xte[:, idx]
        res["E4"][name] = dict(
            acc=accuracy(m, xs, yte),
            inf=min_eps(m, xs, yte, "inf").numpy(),
            l2=min_eps(m, xs, yte, "2").numpy(),
        )
    # Bayes references for the same subsets
    for name, idx in [("Bayes: signal only", [0]), ("Bayes: weak only", [1, 2, 3, 4, 5]),
                      ("Bayes: all", [0, 1, 2, 3, 4, 5])]:
        mask = torch.zeros(d)
        mask[idx] = 1
        wm = w * mask
        res["E4"][name] = dict(
            acc=((xte @ wm > 0).long() == yte).float().mean().item(),
            inf=bayes_min_eps(xte, yte, wm, "inf").numpy(),
            l2=bayes_min_eps(xte, yte, wm, "2").numpy(),
        )
    res["L1"] = {}
    for lam in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]:
        torch.manual_seed(seed)
        m = train_mlp(xtr, ytr, l1=lam)
        cn = first_layer_col_norms(m)
        res["L1"][lam] = dict(
            acc=accuracy(m, xte, yte),
            inf=summ(min_eps(m, xte, yte, "inf")),
            l2=summ(min_eps(m, xte, yte, "2")),
            ratio=(cn[0] / cn[1:].mean()).item(),
        )
    return res


if __name__ == "__main__":
    import sys
    results = dict(e1=[], main=[])
    for s in SEEDS:
        print("seed", s, flush=True)
        results["e1"].append(e1_seed(s))
        results["main"].append(e2_e3_e4_seed(s))
        pickle.dump(results, open("root_cause_results.pkl", "wb"))
    print("done")
