"""Larger-network versions of the real-data checks: a deep wide MLP on MNIST, ResNet-18 on CIFAR-10.
Run: python study/real_big.py r6_big r5_big r3_big r4a_big r9_big   (env SMOKE=1 for a tiny test)
Results -> study/results/big_<name>.pkl"""
import os, pickle, sys, math
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/akayyam/adv_reason"); sys.path.insert(0, "/home/akayyam/adv_reason/study")
import real_checks as rcx
from real_checks import load, subset, flip_size, med, pca_projector, dct_projectors, HI, DEV


def accuracy(m, x, y, bs=500):
    c = 0
    with torch.no_grad():
        for i in range(0, len(x), bs): c += (m(x[i:i + bs].to(DEV)).argmax(1) == y[i:i + bs].to(DEV)).sum().item()
    return c / len(x)
OUT = "/home/akayyam/adv_reason/study/results/"
SMOKE = bool(os.environ.get("SMOKE"))


def save(n, o): pickle.dump(o, open(OUT + f"big_{n}.pkl", "wb"))


class Block(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(o); self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(o)
        self.sc = nn.Sequential() if stride == 1 and i == o else nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))
    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x))); y = self.b2(self.c2(y)); return torch.relu(y + self.sc(x))


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        cfg = [(64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 512, 2), (512, 512, 1)]
        self.blocks = nn.Sequential(*[Block(*c) for c in cfg]); self.fc = nn.Linear(512, 10)
    def forward(self, x):
        x = self.blocks(self.stem(x.view(-1, 3, 32, 32))); return self.fc(x.mean((2, 3)))


def big_mlp(d): return nn.Sequential(nn.Linear(d, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 10))


def train_big(name, x, y, seed, wd=None, in_dim=None):
    torch.manual_seed(seed); x, y = x.to(DEV), y.to(DEV)
    if name == "cifar":
        m = ResNet18().to(DEV); wd_ = 5e-4 if wd is None else wd; lr = 0.1; spe = (len(x) + 127) // 128; ep = 1 if SMOKE else max(15, 1500 // spe)
    else:
        m = big_mlp(in_dim or 784).to(DEV); wd_ = 1e-4 if wd is None else wd; lr = 0.05; spe = (len(x) + 127) // 128; ep = 1 if SMOKE else max(20, 1000 // spe)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=wd_, nesterov=True)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=ep * spe)
    for _ in range(ep):
        m.train(); p = torch.randperm(len(x), device=DEV)
        for i in range(0, len(x), 128):
            b = p[i:i + 128]
            if len(b) < 2: continue
            opt.zero_grad(); nn.functional.cross_entropy(m(x[b]), y[b]).backward(); opt.step(); sch.step()
    return m.eval()


def S(seeds_default): return 1 if SMOKE else seeds_default


def r6_big():
    res = {}
    for name, ns, seeds in [("mnist", [100, 1000, 10000, 60000], 3), ("cifar", [1000, 5000, 20000], 2)]:
        xtr, ytr, xte, yte = load(name); hi_inf, hi_2 = HI[name]
        for n in (ns[:2] if SMOKE else ns):
            row = {"acc": [], "inf": [], "l2": []}
            for s in range(S(seeds)):
                x, y = subset(xtr, ytr, n, s); m = train_big(name, x, y, s)
                row["acc"].append(accuracy(m, xte, yte)); row["inf"].append(med(flip_size(m, xte, yte, "inf", hi_inf))); row["l2"].append(med(flip_size(m, xte, yte, "2", hi_2)))
            res[(name, n)] = {k: float(np.mean(v)) for k, v in row.items()}
            print("r6_big", name, n, {k: round(v, 4) for k, v in res[(name, n)].items()}, flush=True)
    save("r6", res)


def r5_big():
    res = {}
    for name, k, wds, n in [("mnist", 50, [1e-4, 0.0], 10000), ("cifar", 200, [5e-4, 0.0], 20000)]:
        xtr, ytr, xte, yte = load(name); hi_inf, hi_2 = HI[name]
        for wd in wds:
            row = {"tangent": [], "normal": [], "full": [], "acc": []}
            for s in range(S(2)):
                x, y = subset(xtr, ytr, n, s); P = pca_projector(x, k); Pn = torch.eye(P.shape[0], device=DEV) - P
                m = train_big(name, x, y, s, wd=wd)
                row["acc"].append(accuracy(m, xte, yte)); row["tangent"].append(med(flip_size(m, xte, yte, "2", hi_2, P=P)))
                row["normal"].append(med(flip_size(m, xte, yte, "2", hi_2, P=Pn))); row["full"].append(med(flip_size(m, xte, yte, "2", hi_2)))
            res[(name, wd)] = {kk: float(np.mean(v)) for kk, v in row.items()}
            print("r5_big", name, wd, {kk: round(v, 3) for kk, v in res[(name, wd)].items()}, flush=True)
    save("r5", res)


def r3_big():
    res = {}
    for name, h, ch, n in [("mnist", 28, 1, 10000), ("cifar", 32, 3, 20000)]:
        xtr, ytr, xte, yte = load(name); hi_inf, hi_2 = HI[name]; P1 = dct_projectors(h, ch); Ps = {}
        for tag, p1 in P1.items():
            p1 = torch.tensor(p1, dtype=torch.float32); Ps[tag] = (p1 if ch == 1 else torch.block_diag(*[p1] * ch)).to(DEV)
        row = {"low": [], "high": [], "full": [], "acc": []}
        for s in range(S(2)):
            x, y = subset(xtr, ytr, n, s); m = train_big(name, x, y, s)
            row["acc"].append(accuracy(m, xte, yte)); row["low"].append(med(flip_size(m, xte, yte, "2", hi_2, P=Ps["low"]))); row["high"].append(med(flip_size(m, xte, yte, "2", hi_2, P=Ps["high"])))
            row["full"].append(med(flip_size(m, xte, yte, "2", hi_2)))
        res[name] = {k: float(np.mean(v)) for k, v in row.items()}
        print("r3_big", name, {k: round(v, 3) for k, v in res[name].items()}, flush=True)
    save("r3", res)


def r4a_big():
    res = {}
    xtr, ytr, xte, yte = load("mnist"); hi_inf, _ = HI["mnist"]; row = {k: [] for k in ["base_acc", "noise_acc", "base_inf", "real_only", "noise_only", "all"]}
    for s in range(S(3)):
        x, y = subset(xtr, ytr, 10000, s); g = torch.Generator().manual_seed(100 + s)
        xn = torch.cat([x, 0.3 * torch.randn(len(x), 784, generator=g)], 1); xten = torch.cat([xte, 0.3 * torch.randn(len(xte), 784, generator=g)], 1)
        m0 = train_big("mnist", x, y, s); m1 = train_big("mnist", xn, y, s, in_dim=1568)
        row["base_acc"].append(accuracy(m0, xte, yte)); row["noise_acc"].append(accuracy(m1, xten, yte)); row["base_inf"].append(med(flip_size(m0, xte, yte, "inf", hi_inf)))
        mr = torch.cat([torch.ones(784), torch.zeros(784)]).to(DEV); mn = 1 - mr
        row["real_only"].append(med(flip_size(m1, xten, yte, "inf", hi_inf, mask=mr))); row["noise_only"].append(med(flip_size(m1, xten, yte, "inf", hi_inf, mask=mn)))
        row["all"].append(med(flip_size(m1, xten, yte, "inf", hi_inf)))
    res["mnist"] = {k: float(np.mean(v)) for k, v in row.items()}; print("r4a_big", res["mnist"], flush=True); save("r4a", res)


def r9_big():
    res = {}
    for name, arch_n in [("mnist", 10000), ("cifar", 20000)]:
        xtr, ytr, xte, yte = load(name); hi_inf, hi_2 = HI[name]; row = {"acc": [], "inf": [], "l2": []}
        for s in range(S(2)):
            x, y = subset(xtr, ytr, arch_n, s); m = train_big(name, x, y, s)
            row["acc"].append(accuracy(m, xte, yte)); row["inf"].append(med(flip_size(m, xte, yte, "inf", hi_inf))); row["l2"].append(med(flip_size(m, xte, yte, "2", hi_2)))
        res[name] = {k: float(np.mean(v)) for k, v in row.items()}; print("r9_big", name, {k: round(v, 4) for k, v in res[name].items()}, flush=True)
    save("r9", res)


if __name__ == "__main__":
    for n in sys.argv[1:]: globals()[n]()
