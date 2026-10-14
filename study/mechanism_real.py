"""Direct real-data test of the accumulation mechanism. For many trained models measure
  eps_inf (per-input flip size), eps_2 (total flip size), and S = median ||g||_1/||g||_2 of the loss gradient at clean points.
Linear picture: eps_inf ~ eps_2 / S.  We test the prediction across models and decompose log eps_inf = log eps_2 - log S.
Results -> study/results/mechanism_real.pkl"""
import pickle, sys, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, "/home/akayyam/adv_reason"); sys.path.insert(0, "/home/akayyam/adv_reason/study")
import real_checks as rcx
import root_cause_cifar as rc
from real_checks import load, subset, flip_size, HI, DEV
OUT = "/home/akayyam/adv_reason/study/results/"


def grad_spread(model, x, y, n=300):
    x, y = x[:n].to(DEV), y[:n].to(DEV)
    with torch.no_grad(): ok = model(x).argmax(1) == y
    xq = x[ok].clone().requires_grad_(True); Fn.cross_entropy(model(xq), y[ok], reduction="sum").backward()
    g = xq.grad.flatten(1); r = g.abs().sum(1) / g.norm(dim=1).clamp_min(1e-12); return float(r.median())


def mlp(d, widths): 
    layers, i = [], d
    for w in widths: layers += [nn.Linear(i, w), nn.ReLU()]; i = w
    return nn.Sequential(*layers, nn.Linear(i, 10)).to(DEV)


class NetRes(rc.Net):
    def __init__(self, res): super().__init__(); self.res = res
    def forward(self, x): return self.f(x.view(-1, 3, self.res, self.res))


def train(model, x, y, wd, lr, seed, ep_min=15):
    torch.manual_seed(seed); x, y = x.to(DEV), y.to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    spe = (len(x) + 127) // 128; ep = max(ep_min, 1500 // spe); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=ep * spe)
    for _ in range(ep):
        model.train(); p = torch.randperm(len(x), device=DEV)
        for i in range(0, len(x), 128):
            b = p[i:i + 128]
            if len(b) < 2: continue
            opt.zero_grad(); Fn.cross_entropy(model(x[b]), y[b]).backward(); opt.step(); sch.step()
    return model.eval()


def measure(model, xte, yte, hi_inf, hi_2, tag, d, extra):
    with torch.no_grad(): acc = float((model(xte[:1000].to(DEV)).argmax(1) == yte[:1000].to(DEV)).float().mean())
    ei = float(np.median(flip_size(model, xte, yte, "inf", hi_inf))); e2 = float(np.median(flip_size(model, xte, yte, "2", hi_2)))
    S = grad_spread(model, xte, yte); rec = dict(tag=tag, d=d, acc=acc, eps_inf=ei, eps_2=e2, S_grad=S, S_flip=e2 / ei, **extra)
    print(rec, flush=True); return rec


if __name__ == "__main__":
    recs = []
    for name in ["mnist", "fmnist"]:
        xtr, ytr, xte, yte = load(name); hi_inf, hi_2 = HI[name]
        for widths, wd, n in itertools.product([[64], [256], [1024], [256, 256], [1024, 1024, 1024]], [0.0, 1e-4, 1e-3], [1000, 10000]):
            x, y = subset(xtr, ytr, n, 0); m = train(mlp(784, widths), x, y, wd, 0.05, 0)
            recs.append(measure(m, xte, yte, hi_inf, hi_2, name, 784, dict(arch=str(widths), wd=wd, n=n)))
    xtr, ytr, xte, yte = load("cifar"); hi_inf, hi_2 = HI["cifar"]
    def down(x, res): return Fn.adaptive_avg_pool2d(x.view(-1, 3, 32, 32), res).flatten(1) if res != 32 else x
    for res, wd, n, seed in itertools.product([8, 16, 32], [0.0, 5e-4], [5000, 20000], [0, 1]):
        x, y = subset(xtr, ytr, n, seed); xr, xtr_ = down(x, res), down(xte, res)
        m = train(NetRes(res).to(DEV), xr, y, wd, 0.05, seed)
        recs.append(measure(m, xtr_, yte, hi_inf, hi_2, "cifar_cnn", 3 * res * res, dict(arch=f"cnn_res{res}", wd=wd, n=n, seed=seed)))
    for wd, n, seed in itertools.product([0.0, 5e-4], [5000, 20000], [0, 1]):
        x, y = subset(xtr, ytr, n, seed); m = train(mlp(3072, [512]), x, y, wd, 0.02, seed)
        recs.append(measure(m, xte, yte, hi_inf, hi_2, "cifar_mlp", 3072, dict(arch="mlp512", wd=wd, n=n, seed=seed)))
    pickle.dump(recs, open(OUT + "mechanism_real.pkl", "wb"))
