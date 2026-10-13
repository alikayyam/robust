"""Toy experiments for the root-cause study (see study/PLAN.md). Run: python study/toy_experiments.py h10 h6 ...
All tasks: independent Gaussian clues x_i = (y-1/2)*dmu_i + sigma_i*xi, unclipped, so the Bayes rule is exactly linear.
Results -> study/results/toy_<name>.pkl"""
import math, pickle, sys
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.stats import norm as N01
from sklearn.svm import SVC
sys.path.insert(0, "/home/akayyam/adv_reason")
import root_cause_experiments as rce

torch.set_num_threads(1)
SEEDS = list(range(10))
OUT = "/home/akayyam/adv_reason/study/results/"


def gen_for(seed, tag):
    g = torch.Generator(); g.manual_seed(1000 * seed + tag); return g


def save(name, obj):
    pickle.dump(obj, open(OUT + f"toy_{name}.pkl", "wb"))


def med(a):
    a = np.asarray(a); return float(np.median(a)) if len(a) else float("nan")


# ------------------------------------------------------------------ H10: inevitable ceiling
def robust_acc_linear(w, dmu, sigma, eps):
    """accuracy under the worst l_inf perturbation of size eps for the linear rule w (exact)."""
    w = np.asarray(w, float)
    mean = 0.5 * (w * dmu).sum() - eps * np.abs(w).sum()
    sd = math.sqrt(((w * sigma) ** 2).sum())
    return N01.cdf(mean / sd)


def run_h10():
    dmu = np.array([1.0] + [0.4] * 5); sigma = np.array([0.3] + [0.25] * 5)
    eps_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    bayes_w = dmu / sigma ** 2
    res = {"eps": eps_grid, "bayes": [], "best_linear": [], "best_theta": [], "reliable_only": [], "mlp": [], "mlp_clean": []}
    thetas = np.linspace(0, math.pi / 2, 2001)
    for e in eps_grid:
        res["bayes"].append(robust_acc_linear(bayes_w, dmu, sigma, e))
        res["reliable_only"].append(robust_acc_linear([1, 0, 0, 0, 0, 0], dmu, sigma, e))
        accs = [robust_acc_linear([math.cos(t)] + [math.sin(t)] * 5, dmu, sigma, e) for t in thetas]
        j = int(np.argmax(accs)); res["best_linear"].append(accs[j]); res["best_theta"].append(float(thetas[j]))
    mlp = {e: [] for e in eps_grid}
    for s in SEEDS:
        g = gen_for(s, 1); xt, yt = rce.make_data(4000, torch.tensor(dmu).float(), torch.tensor(sigma).float(), g)
        xs, ys = rce.make_data(2000, torch.tensor(dmu).float(), torch.tensor(sigma).float(), g)
        torch.manual_seed(s); m = rce.train_mlp(xt, yt)
        with torch.no_grad(): ok = m(xs).argmax(1) == ys
        r = np.full(len(xs), -1.0); r[ok.numpy()] = rce.min_eps(m, xs, ys, "inf").numpy()
        for e in eps_grid: mlp[e].append(float((r > e).mean()) if e > 0 else float(ok.float().mean()))
        print("h10 seed", s, flush=True)
    res["mlp"] = [float(np.mean(mlp[e])) for e in eps_grid]; res["mlp_sd"] = [float(np.std(mlp[e], ddof=1)) for e in eps_grid]
    save("h10", res); return res


# ------------------------------------------------------------------ H6: data quantity (standard training only)
def spread_task(k, dprime=1 / 0.3, sigma=0.25):
    return torch.full((k,), dprime * sigma / math.sqrt(k)), torch.full((k,), sigma)


def run_h6():
    k = 16; dmu, sigma = spread_task(k); w = rce.bayes_weights(dmu, sigma)
    ns = [30, 60, 120, 250, 500, 1000, 4000]
    res = {"ns": ns, "k": k, "acc": {n: [] for n in ns}, "inf": {n: [] for n in ns}, "l2": {n: [] for n in ns}, "bayes_inf": [], "bayes_l2": [], "wnorm_ratio": {n: [] for n in ns}}
    for s in SEEDS:
        g = gen_for(s, 2); xs, ys = rce.make_data(2000, dmu, sigma, g)
        res["bayes_inf"].append(med(rce.bayes_min_eps(xs, ys, w, "inf").numpy())); res["bayes_l2"].append(med(rce.bayes_min_eps(xs, ys, w, "2").numpy()))
        for n in ns:
            xt, yt = rce.make_data(n, dmu, sigma, g)
            torch.manual_seed(s); m = rce.train_mlp(xt, yt)
            res["acc"][n].append(rce.accuracy(m, xs, ys))
            res["inf"][n].append(med(rce.min_eps(m, xs, ys, "inf").numpy())); res["l2"][n].append(med(rce.min_eps(m, xs, ys, "2", hi=8.0).numpy()))
        print("h6 seed", s, flush=True)
    save("h6", res); return res


# ------------------------------------------------------------------ H8: random noise vs worst case
def run_h8():
    ks = [1, 2, 4, 8, 16, 32, 64]
    res = {"ks": ks, "adv": [], "rand": [], "ratio": [], "pred_ratio": []}
    rng = np.random.default_rng(0)
    for k in ks:
        dmu, sigma = spread_task(k); w = (dmu / sigma ** 2).numpy().astype(float)
        advs, rands = [], []
        for s in SEEDS:
            g = gen_for(s, 3); x, y = rce.make_data(500, dmu, sigma, g)
            llr = (x.numpy() @ w); pred = llr > 0; ok = pred == y.numpy().astype(bool)
            llr = llr[ok]; m = np.abs(llr)
            adv = m / np.abs(w).sum()
            U = rng.choice([-1.0, 1.0], size=(200, k))                  # random sign noise, l_inf size s
            a = -np.sign(llr)[:, None] * (U @ w)[None, :]                # push toward the wrong side
            with np.errstate(divide="ignore"):
                thr = np.where(a > 0, m[:, None] / np.maximum(a, 1e-12), np.inf)
            rnd = np.percentile(thr, 25, axis=1)                          # size at which a quarter of random draws flip
            advs.append(np.median(adv)); rands.append(np.median(rnd))
        res["adv"].append(float(np.mean(advs))); res["rand"].append(float(np.mean(rands)))
        res["ratio"].append(res["rand"][-1] / res["adv"][-1])
        res["pred_ratio"].append(float(np.abs(w).sum() / np.linalg.norm(w) / 0.6745))   # large-k prediction: ||w||1 / (0.674 ||w||2)
        print("h8", k, res["adv"][-1], res["rand"][-1], flush=True)
    save("h8", res); return res


# ------------------------------------------------------------------ H4a: sparse vs dense signal, same total evidence
def run_h4a():
    d = 64; dprime = 1 / 0.3; sg = 0.25
    ss = [1, 2, 4, 8, 16, 32, 64]
    res = {"ss": ss, "d": d, "bayes": [], "mlp": [], "mlp_acc": [], "mlp_l2": [], "bayes_l2": []}
    for s_ in ss:
        dmu = torch.zeros(d); dmu[:s_] = dprime * sg / math.sqrt(s_); sigma = torch.full((d,), sg); w = rce.bayes_weights(dmu, sigma)
        b, bl, mi, ml, ma = [], [], [], [], []
        for s in SEEDS:
            g = gen_for(s, 4); xt, yt = rce.make_data(4000, dmu, sigma, g); xs, ys = rce.make_data(2000, dmu, sigma, g)
            b.append(med(rce.bayes_min_eps(xs, ys, w, "inf").numpy())); bl.append(med(rce.bayes_min_eps(xs, ys, w, "2").numpy()))
            torch.manual_seed(s); m = rce.train_mlp(xt, yt)
            ma.append(rce.accuracy(m, xs, ys)); mi.append(med(rce.min_eps(m, xs, ys, "inf").numpy())); ml.append(med(rce.min_eps(m, xs, ys, "2", hi=8.0).numpy()))
        res["bayes"].append(float(np.mean(b))); res["bayes_l2"].append(float(np.mean(bl))); res["mlp"].append(float(np.mean(mi)))
        res["mlp_l2"].append(float(np.mean(ml))); res["mlp_acc"].append(float(np.mean(ma)))
        print("h4a", s_, res["bayes"][-1], res["mlp"][-1], flush=True)
    save("h4a", res); return res


# ------------------------------------------------------------------ H5: off-manifold directions
def sub_radius(model, x, y, P, hi=8.0, iters=14, steps=20):
    """minimal l_2 flip radius when the perturbation is restricted to the subspace with projector P."""
    with torch.no_grad(): ok = model(x).argmax(1) == y
    x, y = x[ok], y[ok]
    def flips(eps):
        delta = torch.zeros_like(x); fl = torch.zeros(len(x), dtype=torch.bool); step = eps / 4
        for _ in range(steps):
            d = delta.clone().requires_grad_(True); lg = model(x + d); fl |= lg.argmax(1) != y
            g = torch.autograd.grad(torch.nn.functional.cross_entropy(lg, y, reduction="sum"), d)[0] @ P
            with torch.no_grad():
                delta = delta + step * g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                delta = delta * torch.clamp(eps / delta.norm(dim=1, keepdim=True).clamp_min(1e-12), max=1.0)
        with torch.no_grad(): fl |= model(x + delta).argmax(1) != y
        return fl
    lo = torch.zeros(len(x), 1); up = torch.full((len(x), 1), hi)
    reach = flips(up)
    for _ in range(iters):
        mid = (lo + up) / 2; f = flips(mid).unsqueeze(1); up = torch.where(f, mid, up); lo = torch.where(f, lo, mid)
    out = up.squeeze(1); out[~reach] = float("inf"); return out.numpy()


def run_h5(wd=1e-2, tag="h5"):
    d, kdim = 64, 4; sig_offs = [0.0, 0.01, 0.03, 0.1, 0.3]
    res = {"sig_offs": sig_offs, "tangent": {}, "normal": {}, "full": {}, "acc": {}, "wratio": {}}
    for so in sig_offs:
        for key in ["tangent", "normal", "full", "acc", "wratio"]: res[key][so] = []
        for s in range(5):
            g = gen_for(s, 5); torch.manual_seed(s)
            Q, _ = torch.linalg.qr(torch.randn(d, kdim, generator=g))
            def mk(n):
                y = torch.randint(0, 2, (n,), generator=g); z = (y.float().unsqueeze(1) - 0.5) * 1.0 + 0.3 * torch.randn(n, kdim, generator=g)
                return z @ Q.T + so * torch.randn(n, d, generator=g), y
            xt, yt = mk(2000); xs, ys = mk(1000)
            m = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
            opt = torch.optim.Adam(m.parameters(), lr=0.02, weight_decay=wd)
            for _ in range(500):
                opt.zero_grad(); torch.nn.functional.cross_entropy(m(xt), yt).backward(); opt.step()
            P = Q @ Q.T; Pn = torch.eye(d) - P
            res["acc"][so].append(rce.accuracy(m, xs, ys))
            res["tangent"][so].append(med(sub_radius(m, xs, ys, P))); res["normal"][so].append(med(sub_radius(m, xs, ys, Pn)))
            res["full"][so].append(med(sub_radius(m, xs, ys, torch.eye(d))))
            W = m[0].weight.data
            res["wratio"][so].append(float((W @ Pn).norm() / (W @ P).norm()))
        print("h5", so, np.mean(res["tangent"][so]), np.mean(res["normal"][so]), flush=True)
    save(tag, res); return res


def run_h5_nowd():
    return run_h5(wd=0.0, tag="h5_wd0")


# ------------------------------------------------------------------ H7: optimiser / objective bias on linear classifiers
def linf_radius(w, x, y):
    marg = y * (x @ w); ok = marg > 0
    return med(marg[ok] / np.abs(w).sum()) if ok.any() else float("nan")


def run_h7():
    d = 100; n = 60; dp = 3.3
    ss = [1, 3, 10, 30, 100]
    res = {"ss": ss, "methods": ["bayes (oracle)", "l2 max-margin (GD limit)", "l_inf max-margin (LP)"], "acc": {}, "rad": {}}
    for s_ in ss:
        for mth in res["methods"]: res["acc"][(s_, mth)] = []; res["rad"][(s_, mth)] = []
        gap = dp / math.sqrt(s_)
        for sd in SEEDS:
            rng = np.random.default_rng(100 * sd + s_)
            def mk(m):
                y = rng.integers(0, 2, m) * 2 - 1
                x = rng.standard_normal((m, d)); x[:, :s_] += (y[:, None] * 0.5) * gap
                return x, y
            xt, yt = mk(n); xs, ys = mk(2000)
            ws = {}
            w = np.zeros(d); w[:s_] = gap; ws["bayes (oracle)"] = w
            svm = SVC(kernel="linear", C=1e6).fit(xt, yt); ws["l2 max-margin (GD limit)"] = svm.coef_[0]
            # l_inf max-margin: min ||w||_1 s.t. y_i w.x_i >= 1  (w = u - v)
            c = np.ones(2 * d); A = -(yt[:, None] * np.hstack([xt, -xt])); b = -np.ones(n)
            lp = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
            ws["l_inf max-margin (LP)"] = lp.x[:d] - lp.x[d:] if lp.success else np.zeros(d)
            for mth, w in ws.items():
                res["acc"][(s_, mth)].append(float(((xs @ w > 0) * 2 - 1 == ys).mean())); res["rad"][(s_, mth)].append(linf_radius(w, xs, ys))
        print("h7", s_, {m: (round(np.mean(res["acc"][(s_, m)]), 3), round(np.nanmean(res["rad"][(s_, m)]), 3)) for m in res["methods"]}, flush=True)
    save("h7", res); return res


if __name__ == "__main__":
    for name in sys.argv[1:]: globals()["run_" + name]()
