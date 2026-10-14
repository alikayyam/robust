"""Real-data counterparts of the toy hypotheses (MNIST, Fashion-MNIST, CIFAR-10). Run: python study/real_checks.py r8 r4a ...
Same yardstick as the toy study: minimal flip size (per-input l_inf / total l_2) over correctly classified test points. Results -> study/results/real_<name>.pkl"""
import math, pickle, sys
import numpy as np, torch, torch.nn as nn
from scipy.optimize import linprog
from scipy.fft import dct
from sklearn.svm import SVC
from torchvision import datasets
sys.path.insert(0, "/home/akayyam/adv_reason")
import root_cause_realdata as rd
import root_cause_cifar as rc

DEV = rd.DEV; OUT = "/home/akayyam/adv_reason/study/results/"
NTEST = 300


def save(name, obj): pickle.dump(obj, open(OUT + f"real_{name}.pkl", "wb"))
def med(a): a = np.asarray(a); a = a[np.isfinite(a)]; return float(np.median(a)) if len(a) else float("nan")


# ------------------------------------------------------------------ data / models
def load(name):
    if name == "mnist":
        tr, te = rd.load(); return tr[0], tr[1], te[0], te[1]
    if name == "fmnist":
        f = lambda d: (d.data.float().div(255).view(-1, 784) - 0.5, d.targets)
        tr = datasets.FashionMNIST("/home/akayyam/data", train=True, download=False); te = datasets.FashionMNIST("/home/akayyam/data", train=False, download=False)
        a, b = f(tr), f(te); return a[0], a[1], b[0], b[1]
    if name == "cifar":
        tr, te = rc.load(); return tr[0], tr[1], te[0], te[1]


def subset(x, y, n, seed):
    g = torch.Generator().manual_seed(seed); i = torch.randperm(len(x), generator=g)[:n]; return x[i], y[i]


def train_model(name, x, y, seed, wd=None, arch=None):
    torch.manual_seed(seed); x, y = x.to(DEV), y.to(DEV)
    if name in ("mnist", "fmnist") and arch != "cnn":
        old = rd.WD
        if wd is not None: rd.WD = wd
        m = rd.train(x, y, 10); rd.WD = old; return m
    if name in ("mnist", "fmnist"):                       # small CNN for H9
        m = nn.Sequential(View(1, 28), nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32 * 49, 10)).to(DEV)
        opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
        for _ in range(10):
            p = torch.randperm(len(x), device=DEV)
            for i in range(0, len(x), 128):
                b = p[i:i + 128]; opt.zero_grad(); nn.functional.cross_entropy(m(x[b]), y[b]).backward(); opt.step()
        return m.eval()
    if arch == "mlp":                                     # CIFAR MLP
        m = nn.Sequential(nn.Linear(3072, 512), nn.ReLU(), nn.Linear(512, 10)).to(DEV)
        opt = torch.optim.SGD(m.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
        for _ in range(20):
            p = torch.randperm(len(x), device=DEV)
            for i in range(0, len(x), 128):
                b = p[i:i + 128]; opt.zero_grad(); nn.functional.cross_entropy(m(x[b]), y[b]).backward(); opt.step()
        return m.eval()
    m = rc.Net().to(DEV)                                  # CIFAR CNN
    wdc = 5e-4 if wd is None else wd
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=wdc, nesterov=True)
    spe = (len(x) + 127) // 128; ep = 15 if len(x) >= 10000 else max(15, 1500 // spe)   # small sets get a matched step budget
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 0.05, total_steps=ep * spe); m.train()
    for _ in range(ep):
        p = torch.randperm(len(x), device=DEV)
        for i in range(0, len(x), 128):
            b = p[i:i + 128]; opt.zero_grad(); nn.functional.cross_entropy(m(x[b]), y[b]).backward(); opt.step(); sch.step()
    return m.eval()


class View(nn.Module):
    def __init__(self, c, h): super().__init__(); self.c = c; self.h = h
    def forward(self, x): return x.view(-1, self.c, self.h, self.h)


def accuracy(m, x, y):
    with torch.no_grad(): return (m(x.to(DEV)).argmax(1) == y.to(DEV)).float().mean().item()


# ------------------------------------------------------------------ flip sizes with optional restriction
def flips_at(model, x, y, eps, norm, P=None, mask=None, steps=20):
    delta = torch.zeros_like(x); fl = torch.zeros(len(x), dtype=torch.bool, device=x.device); step = eps / 4
    for _ in range(steps):
        d = delta.clone().requires_grad_(True); lg = model(x + d); fl |= lg.argmax(1) != y
        g = torch.autograd.grad(nn.functional.cross_entropy(lg, y, reduction="sum"), d)[0]
        if P is not None: g = g @ P
        if mask is not None: g = g * mask
        with torch.no_grad():
            if norm == "inf": delta = torch.max(torch.min(delta + step * g.sign(), eps), -eps)
            else:
                delta = delta + step * g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                delta = delta * torch.clamp(eps / delta.norm(dim=1, keepdim=True).clamp_min(1e-12), max=1.0)
            if mask is not None: delta = delta * mask
    with torch.no_grad(): fl |= model(x + delta).argmax(1) != y
    return fl


def flip_size(model, x, y, norm, hi, P=None, mask=None, iters=14, n=NTEST):
    x, y = x[:n].to(DEV), y[:n].to(DEV)
    with torch.no_grad(): ok = model(x).argmax(1) == y
    x, y = x[ok], y[ok]
    lo = torch.zeros(len(x), 1, device=DEV); up = torch.full((len(x), 1), hi, device=DEV)
    reach = flips_at(model, x, y, up, norm, P, mask)
    for _ in range(iters):
        mid = (lo + up) / 2; f = flips_at(model, x, y, mid, norm, P, mask).unsqueeze(1); up = torch.where(f, mid, up); lo = torch.where(f, lo, mid)
    out = up.squeeze(1); out[~reach] = float("inf"); return out.cpu().numpy()


HI = {"mnist": (1.0, 20.0), "fmnist": (1.0, 20.0), "cifar": (0.5, 10.0)}       # search ranges (l_inf, l_2)


# ------------------------------------------------------------------ R8: random noise vs worst case (and the linear prediction)
def r8():
    res = {}
    for name in ["mnist", "fmnist", "cifar"]:
        xtr, ytr, xte, yte = load(name); hi_inf, hi_2 = HI[name]; row = {"adv_inf": [], "adv_2": [], "rand_inf": []}
        for s in range(3):
            x, y = subset(xtr, ytr, 10000, s); m = train_model(name, x, y, s)
            ai = flip_size(m, xte, yte, "inf", hi_inf); a2 = flip_size(m, xte, yte, "2", hi_2)
            xs, ys = xte[:NTEST].to(DEV), yte[:NTEST].to(DEV)
            with torch.no_grad(): ok = m(xs).argmax(1) == ys
            xs, ys = xs[ok][:200], ys[ok][:200]
            grid = np.geomspace(0.002, hi_inf, 40); s25 = np.full(len(xs), np.inf)
            for gsz in grid[::-1]:                                  # smallest size at which >= 25% of 40 random sign draws flip
                fr = torch.zeros(len(xs), device=DEV)
                for _ in range(40):
                    u = (torch.randint(0, 2, xs.shape, device=DEV).float() * 2 - 1) * gsz
                    with torch.no_grad(): fr += (m(xs + u).argmax(1) != ys).float()
                hit = (fr / 40 >= 0.25).cpu().numpy(); s25 = np.where(hit, gsz, s25)
            row["adv_inf"].append(med(ai)); row["adv_2"].append(med(a2)); row["rand_inf"].append(med(s25))
        res[name] = {k: float(np.mean(v)) for k, v in row.items()}
        res[name]["ratio_measured"] = res[name]["rand_inf"] / res[name]["adv_inf"]
        res[name]["ratio_predicted"] = res[name]["adv_2"] / res[name]["adv_inf"] / 0.6745
        print("r8", name, {k: round(v, 3) for k, v in res[name].items()}, flush=True)
    save("r8", res)


# ------------------------------------------------------------------ R4a: irrelevant inputs on real data
def r4a():
    res = {}
    for name in ["mnist", "fmnist"]:
        xtr, ytr, xte, yte = load(name); hi_inf, hi_2 = HI[name]; row = {k: [] for k in ["base_acc", "noise_acc", "base_inf", "real_only", "noise_only", "all"]}
        for s in range(3):
            x, y = subset(xtr, ytr, 10000, s); g = torch.Generator().manual_seed(100 + s)
            xn = torch.cat([x, 0.3 * torch.randn(len(x), 784, generator=g)], 1); xten = torch.cat([xte, 0.3 * torch.randn(len(xte), 784, generator=g)], 1)
            m0 = train_model(name, x, y, s)
            torch.manual_seed(s); xt_, yt_ = xn.to(DEV), y.to(DEV)
            old = rd.WD; m1 = nn.Sequential(nn.Linear(1568, 256), nn.ReLU(), nn.Linear(256, 10)).to(DEV)
            opt = torch.optim.SGD(m1.parameters(), lr=0.05, momentum=0.9, weight_decay=rd.WD)
            for _ in range(20):
                p = torch.randperm(len(xt_), device=DEV)
                for i in range(0, len(xt_), 128):
                    b = p[i:i + 128]; opt.zero_grad(); nn.functional.cross_entropy(m1(xt_[b]), yt_[b]).backward(); opt.step()
            m1.eval()
            row["base_acc"].append(accuracy(m0, xte, yte)); row["noise_acc"].append(accuracy(m1, xten, yte))
            row["base_inf"].append(med(flip_size(m0, xte, yte, "inf", hi_inf)))
            mr = torch.cat([torch.ones(784), torch.zeros(784)]).to(DEV); mn = 1 - mr
            row["real_only"].append(med(flip_size(m1, xten, yte, "inf", hi_inf, mask=mr))); row["noise_only"].append(med(flip_size(m1, xten, yte, "inf", hi_inf, mask=mn)))
            row["all"].append(med(flip_size(m1, xten, yte, "inf", hi_inf)))
        res[name] = {k: float(np.mean(v)) for k, v in row.items()}
        print("r4a", name, {k: round(v, 3) for k, v in res[name].items()}, flush=True)
    save("r4a", res)


# ------------------------------------------------------------------ R5: off-manifold (PCA subspace) with and without weight decay
def pca_projector(x, k):
    xc = (x - x.mean(0)).to(DEV); cov = xc.T @ xc / len(xc); ev, V = torch.linalg.eigh(cov); V = V[:, ev.argsort(descending=True)][:, :k]; return V @ V.T


def r5():
    res = {}
    for name, k, wds in [("mnist", 50, [1e-4, 0.0]), ("cifar", 200, [5e-4, 0.0])]:
        xtr, ytr, xte, yte = load(name); hi_inf, hi_2 = HI[name]
        for wd in wds:
            row = {"tangent": [], "normal": [], "full": [], "acc": []}
            for s in range(2 if name == "cifar" else 3):
                x, y = subset(xtr, ytr, 10000, s); P = pca_projector(x, k); Pn = torch.eye(P.shape[0], device=DEV) - P
                m = train_model(name, x, y, s, wd=wd)
                row["acc"].append(accuracy(m, xte, yte)); row["tangent"].append(med(flip_size(m, xte, yte, "2", hi_2, P=P)))
                row["normal"].append(med(flip_size(m, xte, yte, "2", hi_2, P=Pn))); row["full"].append(med(flip_size(m, xte, yte, "2", hi_2)))
            res[(name, wd)] = {kk: float(np.mean(v)) for kk, v in row.items()}
            print("r5", name, wd, {kk: round(v, 3) for kk, v in res[(name, wd)].items()}, flush=True)
    save("r5", res)


# ------------------------------------------------------------------ R3: frequency bands (low vs high) on real images
def dct_projectors(h, ch, frac=0.25):
    C = dct(np.eye(h), norm="ortho", axis=0)                         # (h,h) orthonormal DCT-II matrix (rows = basis)
    B = np.kron(C, C)                                                # 2D basis on h*h pixels
    uu, vv = np.meshgrid(np.arange(h), np.arange(h), indexing="ij"); order = np.argsort((uu + vv).ravel(), kind="stable"); nb = int(frac * h * h)
    out = {}
    for tag, idx in [("low", order[:nb]), ("high", order[-nb:])]:
        M = np.zeros(h * h); M[idx] = 1; P1 = B.T @ np.diag(M) @ B
        P = np.zeros((ch * h * h, ch * h * h))
        for c in range(ch): P[c::ch, c::ch] = P1 if False else 0
        out[tag] = P1
    return out


def r3():
    res = {}
    for name, h, ch in [("mnist", 28, 1), ("cifar", 32, 3)]:
        xtr, ytr, xte, yte = load(name); hi_inf, hi_2 = HI[name]
        P1 = dct_projectors(h, ch)
        Ps = {}
        for tag, p1 in P1.items():
            p1 = torch.tensor(p1, dtype=torch.float32)
            if ch == 1: Ps[tag] = p1.to(DEV)
            else:
                # cifar tensors are flattened as (c,h,w) -> block-diagonal over channels
                Ps[tag] = torch.block_diag(*[p1] * ch).to(DEV)
        row = {"low": [], "high": [], "full": [], "acc": []}
        for s in range(3 if name == "mnist" else 2):
            x, y = subset(xtr, ytr, 10000, s); m = train_model(name, x, y, s)
            row["acc"].append(accuracy(m, xte, yte)); row["low"].append(med(flip_size(m, xte, yte, "2", hi_2, P=Ps["low"]))); row["high"].append(med(flip_size(m, xte, yte, "2", hi_2, P=Ps["high"])))
            row["full"].append(med(flip_size(m, xte, yte, "2", hi_2)))
        res[name] = {k: float(np.mean(v)) for k, v in row.items()}
        print("r3", name, {k: round(v, 3) for k, v in res[name].items()}, flush=True)
    save("r3", res)


# ------------------------------------------------------------------ R6: data quantity (standard training)
name_tag_matched = True


def r6():
    res = {}
    for name, ns, seeds in [("mnist", [100, 300, 1000, 3000, 10000, 30000], 3), ("cifar", [1000, 5000, 20000], 2)]:
        xtr, ytr, xte, yte = load(name); hi_inf, hi_2 = HI[name]
        for n in ns:
            row = {"acc": [], "inf": [], "l2": []}
            for s in range(seeds):
                x, y = subset(xtr, ytr, n, s); m = train_model(name, x, y, s)
                row["acc"].append(accuracy(m, xte, yte)); row["inf"].append(med(flip_size(m, xte, yte, "inf", hi_inf))); row["l2"].append(med(flip_size(m, xte, yte, "2", hi_2)))
            res[(name, n)] = {k: float(np.mean(v)) for k, v in row.items()}
            print("r6", name, n, {k: round(v, 4) for k, v in res[(name, n)].items()}, flush=True)
    save("r6_matched" if name_tag_matched else "r6", res)


# ------------------------------------------------------------------ R7: l2 vs l_inf max-margin linear classifiers, few examples
def r7():
    res = {}
    for name, pair in [("mnist", (3, 7)), ("fmnist", (0, 6))]:
        xtr, ytr, xte, yte = load(name)
        mtr = (ytr == pair[0]) | (ytr == pair[1]); mte = (yte == pair[0]) | (yte == pair[1])
        xa, ya = xtr[mtr].numpy(), ((ytr[mtr] == pair[1]).numpy().astype(int) * 2 - 1); xb, yb = xte[mte].numpy(), ((yte[mte] == pair[1]).numpy().astype(int) * 2 - 1)
        for n in [40, 100, 300]:
            row = {k: [] for k in ["svm_acc", "svm_rad", "lp_acc", "lp_rad"]}
            for s in range(10):
                rng = np.random.default_rng(s); i = rng.permutation(len(xa))[:n]; xt, yt = xa[i], ya[i]
                w = SVC(kernel="linear", C=1e6).fit(xt, yt).coef_[0]
                A = -(yt[:, None] * np.hstack([xt, -xt])); lp = linprog(np.ones(2 * 784), A_ub=A, b_ub=-np.ones(n), bounds=(0, None), method="highs")
                wl = lp.x[:784] - lp.x[784:] if lp.success else np.zeros(784)
                for tag, ww in [("svm", w), ("lp", wl)]:
                    marg = yb * (xb @ ww); ok = marg > 0
                    row[tag + "_acc"].append(float(((xb @ ww > 0) * 2 - 1 == yb).mean())); row[tag + "_rad"].append(med(marg[ok] / np.abs(ww).sum()) if ok.any() else float("nan"))
            res[(name, n)] = {k: float(np.nanmean(v)) for k, v in row.items()}
            print("r7", name, n, {k: round(v, 3) for k, v in res[(name, n)].items()}, flush=True)
    save("r7", res)


# ------------------------------------------------------------------ R2b: Ilyas-style transfer on real data
def targeted_pgd(model, x, t, eps, steps=20):
    delta = torch.zeros_like(x)
    for _ in range(steps):
        d = delta.clone().requires_grad_(True); g = torch.autograd.grad(nn.functional.cross_entropy(model(x + d), t), d)[0]
        delta = (delta - (eps / 4) * g.sign()).clamp(-eps, eps).detach()
    return x + delta


def r2b():
    res = {}
    for name, epss in [("mnist", [0.1, 0.2, 0.3]), ("cifar", [0.03, 0.1])]:
        xtr, ytr, xte, yte = load(name)
        row = {e: {"adv": [], "noise": []} for e in epss}; row["clean_ref"] = []
        for s in range(3 if name == "mnist" else 2):
            x, y = subset(xtr, ytr, 10000, s); m0 = train_model(name, x, y, s); row["clean_ref"].append(accuracy(m0, xte, yte))
            t = torch.randint(0, 10, (len(x),)).to(DEV); xd = x.to(DEV)
            for e in epss:
                xa = torch.cat([targeted_pgd(m0, xd[i:i + 1000], t[i:i + 1000], e).detach() for i in range(0, len(xd), 1000)])
                m1 = train_model(name, xa.cpu(), t.cpu(), s); row[e]["adv"].append(accuracy(m1, xte, yte))
                xn = xd + e * (torch.randint(0, 2, xd.shape, device=DEV).float() * 2 - 1)
                m2 = train_model(name, xn.cpu(), t.cpu(), s); row[e]["noise"].append(accuracy(m2, xte, yte))
        res[name] = {"clean_ref": float(np.mean(row["clean_ref"])), **{e: {k: [float(a) for a in v] for k, v in row[e].items()} for e in epss}}
        print("r2b", name, {e: (round(np.mean(row[e]["adv"]), 3), round(np.mean(row[e]["noise"]), 3)) for e in epss}, flush=True)
    save("r2b", res)


# ------------------------------------------------------------------ R9: architecture on real data
def r9():
    res = {}
    for name, archs in [("mnist", ["mlp", "cnn"]), ("fmnist", ["mlp", "cnn"]), ("cifar", ["mlp", "cnn"])]:
        xtr, ytr, xte, yte = load(name); hi_inf, hi_2 = HI[name]
        for arch in archs:
            row = {"acc": [], "inf": [], "l2": []}
            for s in range(3 if name != "cifar" else 2):
                x, y = subset(xtr, ytr, 10000, s); m = train_model(name, x, y, s, arch=arch)
                row["acc"].append(accuracy(m, xte, yte)); row["inf"].append(med(flip_size(m, xte, yte, "inf", hi_inf))); row["l2"].append(med(flip_size(m, xte, yte, "2", hi_2)))
            res[(name, arch)] = {k: float(np.mean(v)) for k, v in row.items()}
            print("r9", name, arch, {k: round(v, 4) for k, v in res[(name, arch)].items()}, flush=True)
    save("r9", res)


if __name__ == "__main__":
    for n in sys.argv[1:]: globals()[n]()
