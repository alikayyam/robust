"""H9: architecture / invariance. Label = is a small known pattern present ANYWHERE in a 16x16 noisy image? (translation-invariant task).
Compare an MLP, a small translation-invariant CNN (conv + global max), and the hand-built matched filter (the 'right' model), at similar accuracy.
Results -> study/results/invariance.pkl"""
import pickle, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/akayyam/adv_reason")
import root_cause_experiments as rce
torch.set_num_threads(1)
H = 16; D = H * H; SIG = 1.0; K = 3
OUT = "/home/akayyam/adv_reason/study/results/"
TMPL = torch.ones(1, 1, K, K) / K          # unit l2 norm: 9 * (1/3)^2 = 1


def make(n, amp, gen):
    y = torch.randint(0, 2, (n,), generator=gen)
    x = SIG * torch.randn(n, 1, H, H, generator=gen)
    pos = torch.randint(0, H - K + 1, (n, 2), generator=gen)
    for i in range(n):
        if y[i] == 1: x[i, 0, pos[i, 0]:pos[i, 0] + K, pos[i, 1]:pos[i, 1] + K] += amp * TMPL[0, 0]
    return x.view(n, D), y


class MLP(nn.Module):
    def __init__(self): super().__init__(); self.n = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x): return self.n(x)


class CNNg(nn.Module):
    def __init__(self): super().__init__(); self.c = nn.Conv2d(1, 8, K); self.l = nn.Linear(8, 2)
    def forward(self, x): return self.l(torch.relu(self.c(x.view(-1, 1, H, H))).amax((2, 3)))


class Matched(nn.Module):
    def __init__(self, tau): super().__init__(); self.tau = tau
    def forward(self, x):
        s = nn.functional.conv2d(x.view(-1, 1, H, H), TMPL).amax((2, 3)).squeeze(1)
        return torch.stack([torch.zeros_like(s), s - self.tau], 1)


def train(m, x, y, steps=400, lr=0.02, wd=1e-3):
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); nn.functional.cross_entropy(m(x), y).backward(); opt.step()
    return m.eval()


def acc(m, x, y):
    with torch.no_grad(): return (m(x).argmax(1) == y).float().mean().item()


def med(a): a = np.asarray(a); return float(np.median(a)) if len(a) else float("nan")


if __name__ == "__main__":
    amps = [4.0, 5.0, 6.0]; res = {"amps": amps, "rows": {}}
    for amp in amps:
        row = {k: [] for k in ["mlp_acc", "mlp_inf", "mlp_l2", "cnn_acc", "cnn_inf", "cnn_l2", "mf_acc", "mf_inf", "mf_l2"]}
        for s in range(3):
            g = torch.Generator(); g.manual_seed(s)
            xt, yt = make(3000, amp, g); xs, ys = make(600, amp, g)
            torch.manual_seed(s); mlp = train(MLP(), xt, yt)
            torch.manual_seed(s); cnn = train(CNNg(), xt, yt)
            taus = np.linspace(1.5, amp, 40); best = max(taus, key=lambda t: acc(Matched(float(t)), xt, yt)); mf = Matched(float(best)).eval()
            for name, m in [("mlp", mlp), ("cnn", cnn), ("mf", mf)]:
                row[name + "_acc"].append(acc(m, xs, ys))
                row[name + "_inf"].append(med(rce.min_eps(m, xs[:300], ys[:300], "inf").numpy())); row[name + "_l2"].append(med(rce.min_eps(m, xs[:300], ys[:300], "2", hi=20.0).numpy()))
        res["rows"][amp] = {k: float(np.mean(v)) for k, v in row.items()}
        print("h9", amp, {k: round(v, 3) for k, v in res["rows"][amp].items()}, flush=True)
    pickle.dump(res, open(OUT + "invariance.pkl", "wb"))
