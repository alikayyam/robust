"""
E6 (companion to root_cause_paper.tex): the spread mechanism on real data.

MNIST inputs are mixed by a fixed block-diagonal random orthogonal matrix Q_k
(blocks of size k over a random pixel permutation). Q_k preserves l2 geometry
and information exactly; it only changes how widely each direction is spread
across coordinates. An MLP is trained and attacked in the mixed coordinates
with an unclipped ball. Prediction: accuracy and median l2 flip radius are flat
in k, while the median l_inf radius falls (log-log slope about -1/2 until it
saturates). Training uses SGD (rotation-equivariant) with l2 weight decay, not
Adam. Results are written to root_cause_realdata.pkl.
"""

import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KS = [1, 4, 16, 49, 196, 784]
SEEDS = list(range(5))
N_TRAIN, N_TEST = 10000, 1000
HIDDEN, EPOCHS, BS, LR, WD = 256, 20, 128, 0.05, 1e-4
TASKS = {"3v7": (3, 7), "10class": None}


def load():
    tr = datasets.MNIST("/home/akayyam/data", train=True, download=False)
    te = datasets.MNIST("/home/akayyam/data", train=False, download=False)
    f = lambda d: (d.data.float().div(255).view(-1, 784) - 0.5, d.targets)
    return f(tr), f(te)


def select(x, y, classes):
    if classes is None:
        return x, y
    m = (y == classes[0]) | (y == classes[1])
    return x[m], (y[m] == classes[1]).long()


def mixing(k, gen):
    """Block-diagonal orthogonal (784,784) on a random permutation."""
    perm = torch.randperm(784, generator=gen)
    Q = torch.zeros(784, 784)
    for b in range(0, 784, k):
        g, _ = torch.linalg.qr(torch.randn(k, k, generator=gen))
        idx = perm[b:b + k]
        Q[idx.unsqueeze(1), idx.unsqueeze(0)] = g
    return Q


def train(x, y, n_out):
    m = nn.Sequential(nn.Linear(784, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, n_out)).to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.9, weight_decay=WD)
    for _ in range(EPOCHS):
        p = torch.randperm(len(x), device=DEV)
        for i in range(0, len(x), BS):
            b = p[i:i + BS]
            opt.zero_grad()
            nn.functional.cross_entropy(m(x[b]), y[b]).backward()
            opt.step()
    return m.eval()


def flips_at(model, x, y, eps, norm, steps=20):
    delta = torch.zeros_like(x)
    flipped = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    step = eps / 4.0
    for _ in range(steps):
        d = delta.clone().requires_grad_(True)
        lg = model(x + d)
        flipped |= lg.argmax(1) != y
        g = torch.autograd.grad(nn.functional.cross_entropy(lg, y, reduction="sum"), d)[0]
        with torch.no_grad():
            if norm == "inf":
                delta = torch.max(torch.min(delta + step * g.sign(), eps), -eps)
            else:
                delta = delta + step * g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                delta = delta * torch.clamp(eps / delta.norm(dim=1, keepdim=True).clamp_min(1e-12), max=1.0)
    with torch.no_grad():
        flipped |= model(x + delta).argmax(1) != y
    return flipped


def min_eps(model, x, y, norm, hi, iters=14):
    with torch.no_grad():
        ok = model(x).argmax(1) == y
    x, y = x[ok], y[ok]
    lo = torch.zeros(len(x), 1, device=DEV)
    up = torch.full((len(x), 1), hi, device=DEV)
    reach = flips_at(model, x, y, up, norm)
    for _ in range(iters):
        mid = (lo + up) / 2
        f = flips_at(model, x, y, mid, norm).unsqueeze(1)
        up = torch.where(f, mid, up)
        lo = torch.where(f, lo, mid)
    out = up.squeeze(1)
    out[~reach] = float("inf")
    return out.cpu().numpy()


def run(task, seed, tr, te):
    classes = TASKS[task]
    xtr, ytr = select(*tr, classes)
    xte, yte = select(*te, classes)
    g = torch.Generator().manual_seed(seed)
    i = torch.randperm(len(xtr), generator=g)[:N_TRAIN]
    j = torch.randperm(len(xte), generator=g)[:N_TEST]
    xtr, ytr, xte, yte = xtr[i], ytr[i], xte[j], yte[j]
    n_out = 2 if classes else 10
    res = {}
    for k in KS:
        Q = mixing(k, torch.Generator().manual_seed(100 * seed + k))
        a, b = (xtr @ Q.T).to(DEV), (xte @ Q.T).to(DEV)
        torch.manual_seed(seed)
        m = train(a, ytr.to(DEV), n_out)
        yt = yte.to(DEV)
        with torch.no_grad():
            acc = (m(b).argmax(1) == yt).float().mean().item()
        einf = min_eps(m, b, yt, "inf", hi=2.0)
        e2 = min_eps(m, b, yt, "2", hi=20.0)
        res[k] = dict(acc=acc, inf=float(np.median(einf)), l2=float(np.median(e2)))
        print(task, seed, k, res[k], flush=True)
    return res


if __name__ == "__main__":
    tr, te = load()
    out = {t: [run(t, s, tr, te) for s in SEEDS] for t in TASKS}
    pickle.dump(out, open("root_cause_realdata.pkl", "wb"))
