"""H3: shape vs texture / frequency, in a fully controlled synthetic image world (16x16, Gaussian pixel noise).
Cues are unit-norm templates added with a class-dependent sign, so every cue has an exactly known strength (discriminability) and spread.
Exp 1 (2x2): low vs high spatial frequency  x  spread over the image vs concentrated.  Equal strength => equal accuracy.
Exp 2: both a shape-like cue (smooth local blob) and a texture-like cue (full-image checkerboard) present; cue-conflict preference and attack energy.
Results -> study/results/shape_texture.pkl"""
import pickle, sys
import numpy as np, torch, torch.nn as nn
from scipy.stats import norm
sys.path.insert(0, "/home/akayyam/adv_reason")
import root_cause_experiments as rce
torch.set_num_threads(1)
H = 16; D = H * H; SIGMA = 1.0
OUT = "/home/akayyam/adv_reason/study/results/"

X, Y = np.meshgrid(np.arange(H), np.arange(H))
def unit(t): t = t.astype(float).ravel(); return t / np.linalg.norm(t)
blob = np.exp(-((X - 7.5) ** 2 + (Y - 7.5) ** 2) / (2 * 2.0 ** 2))
checker = (-1.0) ** (X + Y)
T = {"low_spread": unit(np.cos(np.pi * (X + 0.5) / H)),      # smooth horizontal gradient over the whole image
     "low_local": unit(blob),                                 # smooth blob in the middle (shape-like)
     "high_spread": unit(checker),                            # full-image checkerboard (texture-like)
     "high_local": unit(checker * blob)}                      # checkerboard confined to the middle
SPREAD = {k: np.abs(v).sum() / np.linalg.norm(v) for k, v in T.items()}   # l1/l2


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Flatten(), nn.Linear(16 * 4 * 4, 2))
    def forward(self, x): return self.f(x.view(-1, 1, H, H))


def mlp(): return nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 2))


def make(n, amps, gen):
    y = torch.randint(0, 2, (n,), generator=gen)
    s = sum(a * torch.tensor(T[k]).float() for k, a in amps.items())
    x = (y.float().unsqueeze(1) - 0.5) * s + SIGMA * torch.randn(n, D, generator=gen)
    return x, y


def train(model, x, y, wd=1e-2, steps=300, lr=0.02):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); nn.functional.cross_entropy(model(x), y).backward(); opt.step()
    return model.eval()


def acc(m, x, y):
    with torch.no_grad(): return (m(x).argmax(1) == y).float().mean().item()


def fgsm_delta(m, x, y, eps):
    xa = x.clone().requires_grad_(True); nn.functional.cross_entropy(m(xa), y).backward()
    return eps * xa.grad.sign()


def med(a): a = np.asarray(a); return float(np.median(a)) if len(a) else float("nan")


def exp1(seeds=3, strength=3.3):
    res = {"spread": SPREAD, "strength": strength, "rows": {}}
    for cue in T:
        row = {k: [] for k in ["bayes_acc", "bayes_inf", "bayes_l2", "mlp_acc", "mlp_inf", "mlp_l2", "cnn_acc", "cnn_inf", "cnn_l2"]}
        w = torch.tensor(strength * T[cue]).float() / SIGMA ** 2
        for s in range(seeds):
            g = torch.Generator(); g.manual_seed(s)
            xt, yt = make(4000, {cue: strength}, g); xs, ys = make(1000, {cue: strength}, g)
            row["bayes_acc"].append(((xs @ w > 0).long() == ys).float().mean().item())
            row["bayes_inf"].append(med(rce.bayes_min_eps(xs, ys, w, "inf").numpy())); row["bayes_l2"].append(med(rce.bayes_min_eps(xs, ys, w, "2").numpy()))
            for name, mk in [("mlp", mlp), ("cnn", CNN)]:
                torch.manual_seed(s); m = train(mk(), xt, yt)
                row[name + "_acc"].append(acc(m, xs, ys))
                row[name + "_inf"].append(med(rce.min_eps(m, xs[:300], ys[:300], "inf").numpy())); row[name + "_l2"].append(med(rce.min_eps(m, xs[:300], ys[:300], "2", hi=20.0).numpy()))
        res["rows"][cue] = {k: float(np.mean(v)) for k, v in row.items()}
        print("exp1", cue, round(SPREAD[cue], 2), {k: round(v, 3) for k, v in res["rows"][cue].items()}, flush=True)
    return res


def exp2(seeds=3):
    conds = [(3.3, 1.0), (2.3, 2.3), (1.0, 3.3), (3.3, 3.3)]   # (shape-like strength, texture-like strength)
    res = {"conds": conds, "rows": {}}
    tl, th = torch.tensor(T["low_local"]).float(), torch.tensor(T["high_spread"]).float()
    for aL, aH in conds:
        row = {k: [] for k in ["acc", "inf", "follow_shape", "bayes_follow_shape", "adv_low_frac", "adv_high_frac"]}
        for s in range(seeds):
            g = torch.Generator(); g.manual_seed(s)
            amps = {"low_local": aL, "high_spread": aH}
            xt, yt = make(4000, amps, g); xs, ys = make(1000, amps, g)
            torch.manual_seed(s); m = train(CNN(), xt, yt)
            row["acc"].append(acc(m, xs, ys)); row["inf"].append(med(rce.min_eps(m, xs[:300], ys[:300], "inf").numpy()))
            # cue conflict: noise-only images + opposing cues (shape says label 1, texture says label 0)
            z = SIGMA * torch.randn(1000, D, generator=g)
            xc = z + 0.5 * aL * tl - 0.5 * aH * th
            with torch.no_grad(): p = m(xc).argmax(1)
            row["follow_shape"].append((p == 1).float().mean().item())
            # best rule's weights are the true ones (aL*tl + aH*th)/SIGMA^2; its conflict preference has this closed form (templates are ~orthonormal)
            row["bayes_follow_shape"].append(float(norm.cdf(0.5 * (aL ** 2 - aH ** 2) / (SIGMA * np.sqrt(aL ** 2 + aH ** 2)))))
            # where does a worst-case l_inf nudge put its energy? project onto each template
            d = fgsm_delta(m, xs, ys, 0.1); dl, dh = (d @ tl).abs(), (d @ th).abs(); tot = d.norm(dim=1)
            row["adv_low_frac"].append((dl / tot).mean().item()); row["adv_high_frac"].append((dh / tot).mean().item())
        res["rows"][(aL, aH)] = {k: float(np.mean(v)) for k, v in row.items()}
        print("exp2", (aL, aH), {k: round(v, 3) for k, v in res["rows"][(aL, aH)].items()}, flush=True)
    return res


if __name__ == "__main__":
    out = {"exp1": exp1(), "exp2": exp2()}
    pickle.dump(out, open(OUT + "shape_texture.pkl", "wb"))
