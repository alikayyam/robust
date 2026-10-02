"""
E6b (companion to root_cause_paper.tex): CNN on CIFAR-10 with the input
projected onto its top-m PCA directions (m in MS). The model sees P_m x
(reshaped to an image); the attacker perturbs raw pixels with an unclipped ball,
so only the in-subspace part of the perturbation matters. Fewer directions =
smaller ||w||_1 at some cost in accuracy (P4). Results: root_cause_cifar.pkl.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from root_cause_realdata import flips_at, min_eps  # noqa: F401  (shared PGD bisection)
import root_cause_realdata as rd

DEV = rd.DEV
MS = [16, 64, 256, 1024, 3072]
SEEDS = list(range(5))
N_TRAIN, N_TEST, EPOCHS, BS, LR, WD = 20000, 500, 15, 128, 0.05, 5e-4


def load():
    tr = datasets.CIFAR10("/home/akayyam/data", train=True, download=False)
    te = datasets.CIFAR10("/home/akayyam/data", train=False, download=False)
    f = lambda d: (torch.tensor(d.data).float().div(255).permute(0, 3, 1, 2).reshape(len(d.data), -1) - 0.5,
                   torch.tensor(d.targets))
    return f(tr), f(te)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        c = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.f = nn.Sequential(*c(3, 32), *c(32, 32), nn.MaxPool2d(2), *c(32, 64), *c(64, 64),
                               nn.MaxPool2d(2), *c(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                               nn.Linear(128, 10))

    def forward(self, x):
        return self.f(x.view(-1, 3, 32, 32))


def run(seed, tr, te):
    g = torch.Generator().manual_seed(seed)
    i = torch.randperm(len(tr[0]), generator=g)[:N_TRAIN]
    j = torch.randperm(len(te[0]), generator=g)[:N_TEST]
    xtr, ytr, xte, yte = tr[0][i], tr[1][i], te[0][j], te[1][j]
    # full eigendecomposition of the 3072x3072 covariance
    cov = xtr.T @ xtr / len(xtr)
    ev, evec = torch.linalg.eigh(cov)
    evec = evec[:, ev.argsort(descending=True)]
    res = {}
    for m in MS:
        P = (evec[:, :m] @ evec[:, :m].T).to(DEV)
        model = nn.Sequential(Fixed(P), Net()).to(DEV)
        torch.manual_seed(seed)
        model[1].apply(lambda l: l.reset_parameters() if hasattr(l, "reset_parameters") else None)
        a, y = xtr.to(DEV), ytr.to(DEV)
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * ((len(a) + BS - 1) // BS))
        model.train()
        for _ in range(EPOCHS):
            p = torch.randperm(len(a), device=DEV)
            for k in range(0, len(a), BS):
                b = p[k:k + BS]
                opt.zero_grad()
                nn.functional.cross_entropy(model(a[b]), y[b]).backward()
                opt.step(); sch.step()
        model.eval()
        xt, yt = xte.to(DEV), yte.to(DEV)
        with torch.no_grad():
            acc = (model(xt).argmax(1) == yt).float().mean().item()
        einf = rd.min_eps(model, xt, yt, "inf", hi=1.0)
        e2 = rd.min_eps(model, xt, yt, "2", hi=20.0)
        res[m] = dict(acc=acc, inf=float(np.median(einf)), l2=float(np.median(e2)))
        print(seed, m, res[m], flush=True)
    return res


class Fixed(nn.Module):
    def __init__(self, P):
        super().__init__()
        self.register_buffer("P", P)

    def forward(self, x):
        return x @ self.P


if __name__ == "__main__":
    tr, te = load()
    out = [run(s, tr, te) for s in SEEDS]
    pickle.dump(out, open("root_cause_cifar.pkl", "wb"))
