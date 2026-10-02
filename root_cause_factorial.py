"""
E5: reliability d' = dmu/sigma  x  scale (class-mean gap dmu), one dimension.

A one-dimensional Gaussian task is built for each (dmu, d') pair with
sigma = dmu / d'. Bayes accuracy depends on d' only; the Ilyas-style
robustness of the feature at budget eps (does the class-conditional
correlation survive an eps perturbation?) depends on dmu vs 2*eps only.
The two are therefore varied independently. Trained models carry a fixed
standardisation layer (mean/std of the training inputs) so that tiny-scale
features are not disadvantaged by the optimiser; attacks act on the raw
input, in raw units.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import root_cause_experiments as R

torch.set_num_threads(1)
DMUS = [0.05, 0.1, 0.2, 0.4, 0.8]
DPRIMES = [1.5, 3.0, 6.0]
SEEDS = R.SEEDS


class Std(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.m, self.s = x.mean(0), x.std(0)
        self.net = R.MLP(x.shape[1])

    def forward(self, x):
        return self.net((x - self.m) / self.s)


def train_std(x, y):
    m = Std(x)
    opt = torch.optim.Adam(m.parameters(), lr=R.LR, weight_decay=R.WD)
    ce = nn.CrossEntropyLoss()
    for _ in range(R.STEPS):
        opt.zero_grad()
        ce(m(x), y).backward()
        opt.step()
    return m


def run():
    out = {}
    for dmu in DMUS:
        for dp in DPRIMES:
            sigma = dmu / dp
            rows = []
            for s in SEEDS:
                g = R.seed_gen(s, int(dmu * 1000 + dp * 10))
                d_, s_ = torch.tensor([dmu]), torch.tensor([sigma])
                xtr, ytr = R.make_data(R.N_TRAIN, d_, s_, g)
                xte, yte = R.make_data(R.N_TEST, d_, s_, g)
                w = R.bayes_weights(d_, s_)
                torch.manual_seed(s)
                m = train_std(xtr, ytr)
                rows.append(dict(
                    acc=R.accuracy(m, xte, yte),
                    bayes_acc=((xte @ w > 0).long() == yte).float().mean().item(),
                    bayes=R.summ(R.bayes_min_eps(xte, yte, w, "inf")),
                    mlp=R.summ(R.min_eps(m, xte, yte, "inf", hi=2.0)),
                ))
            out[(dmu, dp)] = rows
            print(dmu, dp, np.mean([r["bayes"] for r in rows]),
                  np.mean([r["mlp"] for r in rows]), flush=True)
    return out


if __name__ == "__main__":
    pickle.dump(run(), open("root_cause_factorial.pkl", "wb"))
