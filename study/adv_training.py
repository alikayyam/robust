"""Adversarial-training arm (l_inf PGD training). Run: python study/adv_training.py toy_ceiling toy_data real
toy_ceiling: does adversarial training close the gap to the inevitable ceiling of H10?
toy_data:    does data quantity matter once the model is trained adversarially (Schmidt et al.)?
real:        MNIST (MLP) and CIFAR-10 (CNN): standard vs adversarial training at several training-set sizes.
Results -> study/results/at_<name>.pkl"""
import math, pickle, sys
import numpy as np, torch, torch.nn as nn
from scipy.stats import norm as N01
sys.path.insert(0, "/home/akayyam/adv_reason"); sys.path.insert(0, "/home/akayyam/adv_reason/study")
import root_cause_experiments as rce
OUT = "/home/akayyam/adv_reason/study/results/"
SEEDS = list(range(10))


def save(n, o): pickle.dump(o, open(OUT + f"at_{n}.pkl", "wb"))


def pgd_linf(model, x, y, eps, steps, rand_start=True):
    delta = (torch.rand_like(x) * 2 - 1) * eps if rand_start else torch.zeros_like(x)
    for _ in range(steps):
        d = delta.clone().requires_grad_(True)
        g = torch.autograd.grad(nn.functional.cross_entropy(model(x + d), y), d)[0]
        delta = (delta + (2.5 * eps / steps) * g.sign()).clamp(-eps, eps).detach()
    return (x + delta).detach()


def robust_acc(model, x, y, eps, steps=20):
    model.eval()
    xa = pgd_linf(model, x, y, eps, steps)
    with torch.no_grad(): return (model(xa).argmax(1) == y).float().mean().item()


def clean_acc(model, x, y):
    with torch.no_grad(): return (model(x).argmax(1) == y).float().mean().item()


# ---------------------------------------------------------------- toy
def train_toy(x, y, eps, seed, steps=500, wd=1e-2, hidden=8):
    torch.manual_seed(seed); m = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 2))
    opt = torch.optim.Adam(m.parameters(), lr=0.02, weight_decay=wd)
    for _ in range(steps):
        xa = pgd_linf(m, x, y, eps, 5) if eps > 0 else x
        opt.zero_grad(); nn.functional.cross_entropy(m(xa), y).backward(); opt.step()
    return m.eval()


def ceiling(dmu, sigma, eps):
    th = np.linspace(0, math.pi / 2, 1001); best = 0
    for t in th:
        w = np.array([math.cos(t)] + [math.sin(t)] * (len(dmu) - 1))
        best = max(best, N01.cdf((0.5 * (w * dmu).sum() - eps * np.abs(w).sum()) / math.sqrt(((w * sigma) ** 2).sum())))
    return best


def toy_ceiling():
    dmu = torch.tensor([1.0] + [0.4] * 5); sigma = torch.tensor([0.3] + [0.25] * 5)
    res = {"eps": [0.1, 0.2, 0.3], "ceiling": {}, "std": {}, "at": {}, "at_clean": {}, "std_clean": []}
    for e in res["eps"]:
        res["ceiling"][e] = ceiling(dmu.numpy(), sigma.numpy(), e); res["std"][e] = []; res["at"][e] = []; res["at_clean"][e] = []
    for s in SEEDS:
        g = torch.Generator(); g.manual_seed(s); xt, yt = rce.make_data(4000, dmu, sigma, g); xs, ys = rce.make_data(2000, dmu, sigma, g)
        m0 = train_toy(xt, yt, 0.0, s); res["std_clean"].append(clean_acc(m0, xs, ys))
        for e in res["eps"]:
            res["std"][e].append(robust_acc(m0, xs, ys, e)); m1 = train_toy(xt, yt, e, s)
            res["at"][e].append(robust_acc(m1, xs, ys, e)); res["at_clean"][e].append(clean_acc(m1, xs, ys))
        print("toy_ceiling seed", s, {e: (round(res["std"][e][-1], 3), round(res["at"][e][-1], 3)) for e in res["eps"]}, flush=True)
    save("toy_ceiling", res)


def toy_data(k=16, eps=0.05, tag="toy_data", ns=None):
    dprime = 1 / 0.3; sg = 0.25
    dmu = torch.full((k,), dprime * sg / math.sqrt(k)); sigma = torch.full((k,), sg)
    ceil = N01.cdf(math.sqrt(k) * (dmu[0].item() / 2 - eps) / sg)
    ns = ns or [30, 100, 300, 1000, 4000]
    res = {"ns": ns, "k": k, "eps": eps, "ceiling": ceil, "std_clean": {n: [] for n in ns}, "std_rob": {n: [] for n in ns}, "at_clean": {n: [] for n in ns}, "at_rob": {n: [] for n in ns}}
    for s in SEEDS:
        g = torch.Generator(); g.manual_seed(s); xs, ys = rce.make_data(4000, dmu, sigma, g)
        for n in ns:
            xt, yt = rce.make_data(n, dmu, sigma, g)
            m0 = train_toy(xt, yt, 0.0, s); m1 = train_toy(xt, yt, eps, s)
            res["std_clean"][n].append(clean_acc(m0, xs, ys)); res["std_rob"][n].append(robust_acc(m0, xs, ys, eps))
            res["at_clean"][n].append(clean_acc(m1, xs, ys)); res["at_rob"][n].append(robust_acc(m1, xs, ys, eps))
        print(tag, "seed", s, flush=True)
    save(tag, res)


# ---------------------------------------------------------------- real
def toy_data_hi():
    toy_data(k=64, eps=0.03, tag="toy_data_k64", ns=[30, 100, 300, 1000, 4000, 16000])
    toy_data(k=256, eps=0.015, tag="toy_data_k256", ns=[100, 300, 1000, 4000, 16000])


def real():
    import root_cause_realdata as rd, root_cause_cifar as rc
    import real_checks as rcx
    DEV = rd.DEV
    res = {}
    cfgs = [("mnist", 0.1, [100, 1000, 10000], 3, 10, 0.05), ("cifar", 0.0078, [1000, 5000, 20000], 2, 15, 0.05)]
    for name, eps, ns, seeds, ep, lr in cfgs:
        xtr, ytr, xte, yte = rcx.load(name); xte, yte = xte[:1000].to(DEV), yte[:1000].to(DEV)
        for n in ns:
            row = {k: [] for k in ["std_clean", "std_rob", "std_flip", "at_clean", "at_rob", "at_flip"]}
            for s in range(seeds):
                x, y = rcx.subset(xtr, ytr, n, s); x, y = x.to(DEV), y.to(DEV)
                for mode in ["std", "at"]:
                    torch.manual_seed(s)
                    m = (nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)) if name == "mnist" else rc.Net()).to(DEV)
                    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    steps_ep = (len(x) + 127) // 128; ep_ = max(ep, 1500 // steps_ep); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=ep_ * steps_ep)
                    for _ in range(ep_):
                        m.train(); p = torch.randperm(len(x), device=DEV)
                        for i in range(0, len(x), 128):
                            b = p[i:i + 128]; xb = x[b]
                            if mode == "at": m.eval(); xb = pgd_linf(m, xb, y[b], eps, 5); m.train()
                            opt.zero_grad(); nn.functional.cross_entropy(m(xb), y[b]).backward(); opt.step(); sch.step()
                    m.eval()
                    row[mode + "_clean"].append(clean_acc(m, xte, yte)); row[mode + "_rob"].append(robust_acc(m, xte, yte, eps))
                    row[mode + "_flip"].append(float(np.median(rcx.flip_size(m, xte, yte, "inf", 1.0 if name == "mnist" else 0.5))))
            res[(name, n)] = {k: float(np.mean(v)) for k, v in row.items()}; res[(name, n)]["eps"] = eps
            print("at_real", name, n, {k: round(v, 4) for k, v in res[(name, n)].items()}, flush=True)
    save("real", res)


if __name__ == "__main__":
    for n in sys.argv[1:]: globals()[n]()
