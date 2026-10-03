"""New illustration for illustrative_paper.tex: actual margin of one test point
(unclipped task, Model A, seed 0) as FGSM nudges of size 0.3 are applied to
more and more dimensions."""
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import feature_reliance_unclipped as f

EPS = 0.3
(xt, yt), (xs, ys) = f.data(0)
torch.manual_seed(0)
A = f.train(f.MLP(6), xt, yt)


def margin(x, y):
    with torch.no_grad():
        lg = A(x)
        return (lg.gather(1, y[:, None]) - lg.gather(1, (1 - y)[:, None])).squeeze(1)


x = xs.clone().requires_grad_(True)
lg = A(x)
(lg.gather(1, ys[:, None]) - lg.gather(1, (1 - ys)[:, None])).sum().backward()
sgn = -x.grad.sign()                       # direction that lowers the margin
m0 = margin(xs, ys)


def after(dims):
    mask = torch.zeros(6); mask[dims] = 1
    return margin(xs + EPS * sgn * mask, ys)


rel, w5 = after([0]), after([1, 2, 3, 4, 5])
singles = torch.stack([after([d]) for d in range(1, 6)], 1)
cum = torch.stack([after(list(range(1, k + 1))) for k in range(1, 6)], 1)
good = (m0 > 0) & (rel > 0) & (singles > 0).all(1) & (w5 < 0)
idx = torch.where(good)[0]
i = idx[(m0[idx] - m0[idx].median()).abs().argmin()].item()
print("n candidates", len(idx), "margin", m0[i].item(), "reliable-only", rel[i].item(),
      "singles", singles[i].tolist(), "cum", cum[i].tolist())

fig, ax = plt.subplots(figsize=(6.2, 3.6))
ks = np.arange(0, 6)
ax.plot(ks, [m0[i].item()] + cum[i].tolist(), "o-", color="#ff7f0e", label="weak dimensions nudged, one more each step")
ax.plot([1], [rel[i].item()], "s", color="#2ca02c", ms=9, label="only the reliable dimension nudged")
ax.axhline(0, color="k", ls="--", lw=1)
ax.text(5.05, 0.05, "wrong answer below", fontsize=8, ha="right", va="bottom")
ax.set_xlabel("number of weak dimensions nudged (the reliable one is untouched)")
ax.set_ylabel("confidence in the right answer\n(margin; below 0 = fooled)")
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_il_nudges.pdf")

# ---- activation-sparsity strip plot (unclipped task), for illustrative_paper.tex
import pickle
a = pickle.load(open("feature_reliance_unclipped_pgd_actsparse.pkl", "rb"))["actsparse"]
h = pickle.load(open("feature_reliance_unclipped_actsparse_high.pkl", "rb"))
lams = [0.0, 0.3, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
accs = {l: (a[l]["acc"] if l in a else h[l]["acc"]) for l in lams}
fig, ax = plt.subplots(figsize=(6.0, 3.2))
rng = np.random.default_rng(0)
for i, l in enumerate(lams):
    v = np.array(accs[l])
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=18, color="#1f77b4", alpha=0.8)
    ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color="#d62728", lw=2)
ax.set_xticks(range(len(lams))); ax.set_xticklabels([str(l) for l in lams])
ax.set_xlabel("strength of the activity penalty"); ax.set_ylabel("test accuracy")
ax.text(len(lams) - 0.5, 0.52, "dead network", fontsize=8, ha="right")
ax.plot([], [], color="#d62728", lw=2, label="average over 10 runs")
ax.legend(frameon=False, fontsize=8, loc="center left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_il_actsparse.pdf")

# ---- decision slices, example attack, L1 sweep (unclipped task) ----
import torch
(xt0, yt0), (xs0, ys0) = f.data(0)
torch.manual_seed(0)
A0 = f.train(f.MLP(6), xt0, yt0)
torch.manual_seed(0)
B0 = f.train(f.MLP(1), xt0[:, :1], yt0)

fig, axs = plt.subplots(1, 6, figsize=(11, 2.5), sharey=True)
mean = xs0.mean(0)
for d in range(6):
    ax = axs[d]
    lo, hi = (-1.2, 1.2) if d == 0 else (-1.0, 1.0)
    grid = torch.linspace(lo, hi, 200)
    X = mean.repeat(200, 1); X[:, d] = grid
    with torch.no_grad():
        pr = torch.softmax(A0(X), 1)[:, 1]
    ax2 = ax.twinx()
    for c, col in [(0, "#1f77b4"), (1, "#ff7f0e")]:
        ax2.hist(xs0[ys0 == c][:, d].numpy(), bins=25, range=(lo, hi), alpha=0.35, color=col)
    ax2.set_yticks([])
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.plot(grid, pr, "k")
    ax.set_title("reliable $x_0$" if d == 0 else f"weak $x_{d}$", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False); ax2.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("P(label = 1)")
fig.tight_layout(); fig.savefig("fig_il_slices.pdf")

# example attack on one test point where both attacks succeed
EPS = 0.3
def attack(model, x, y, mask):
    xa = x.clone().requires_grad_(True)
    torch.nn.functional.cross_entropy(model(xa), y).backward()
    return (x + EPS * xa.grad.sign() * torch.tensor(mask, dtype=torch.float32)).detach()
with torch.no_grad():
    okA = A0(xs0).argmax(1) == ys0; okB = B0(xs0[:, :1]).argmax(1) == ys0
cand = []
for i in torch.where(okA & okB)[0].tolist():
    xa = attack(A0, xs0[i:i+1], ys0[i:i+1], f.MASKS["weak5"])
    xb = attack(B0, xs0[i:i+1, :1], ys0[i:i+1], [1])
    if A0(xa).argmax(1).item() != ys0[i].item() and B0(xb).argmax(1).item() != ys0[i].item():
        cand.append((i, xa, xb))
i, xa, xb = cand[len(cand) // 2]
x0 = xs0[i]
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw=dict(width_ratios=[6, 1.3]))
w = 0.38
axs[0].bar(np.arange(6) - w/2, x0.numpy(), w, color="#7f7f7f", label="before")
axs[0].bar(np.arange(6) + w/2, xa[0].numpy(), w, color="#ff7f0e", label="after (weak clues nudged by 0.3)")
axs[0].set_xticks(range(6)); axs[0].set_xticklabels(["$x_0$\nreliable"] + [f"$x_{d}$\nweak" for d in range(1, 6)], fontsize=8)
axs[0].set_title("Model A: reliable clue untouched, answer flips", fontsize=9); axs[0].legend(frameon=False, fontsize=8)
axs[1].bar([0 - w/2], [x0[0].item()], w, color="#7f7f7f"); axs[1].bar([0 + w/2], [xb[0, 0].item()], w, color="#2ca02c")
axs[1].set_xticks([0]); axs[1].set_xticklabels(["$x_0$"]); axs[1].set_title("Model B: nudged\nby 0.3, flips", fontsize=9)
for a in axs: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_il_example.pdf")

# L1 sweep
reg = pickle.load(open("feature_reliance_unclipped_reg_phase.pkl", "rb"))["reg"]["L1"]
base = pickle.load(open("feature_reliance_unclipped_reg_phase.pkl", "rb"))["reg"]["L2"][0]
lams = [r["strength"] for r in reg]
fig, ax = plt.subplots(figsize=(5.6, 3.3))
ax.plot(lams, [np.median(r["ratio"]) for r in reg], "o-", color="#1f77b4", label="weight on reliable vs weak clues (median)")
ax.axhline(np.median(base["ratio"]), color="#1f77b4", ls=":", lw=1)
ax.set_xscale("log"); ax.set_xlabel("$L_1$ penalty strength"); ax.set_ylabel("reliable : weak weight ratio", color="#1f77b4")
ax2 = ax.twinx()
ax2.errorbar(lams, [np.mean(r["acc"]) for r in reg], yerr=[np.std(r["acc"], ddof=1) for r in reg], fmt="s--", color="#d62728", capsize=3, label="accuracy (mean ± s.d.)")
ax2.set_ylabel("accuracy", color="#d62728"); ax2.set_ylim(0.7, 1.02)
ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_il_l1sweep.pdf")
print("example idx", i, "n cand", len(cand))

# ================= extra illustrations =================
from matplotlib.patches import Circle, Rectangle
G, O, B_, R = "#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"

# (a) clue histograms: reliable vs one weak vs average of five weak
g = torch.Generator().manual_seed(1)
Xs, Ys = f.make(200000, gen=g)
panels = [("reliable clue $x_0$", Xs[:, 0], 0.0), ("one weak clue $x_1$", Xs[:, 1], 0.0),
          ("average of the five weak clues", Xs[:, 1:].mean(1), 0.0)]
fig, axs = plt.subplots(1, 3, figsize=(10, 2.7), sharey=True)
for ax, (t, v, thr) in zip(axs, panels):
    for c, col in [(0, B_), (1, O)]:
        ax.hist(v[Ys == c].numpy(), bins=60, range=(-1.6, 1.6), density=True, alpha=0.5, color=col)
    accv = (((v > thr).long()) == Ys).float().mean().item()
    ax.set_title(f"{t}\nbest accuracy {accv:.1%}", fontsize=9); ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_il_clues.pdf")

# (b) stacked weights: sum of |w| per group
fig, ax = plt.subplots(figsize=(4.4, 3.4))
ax.bar(0, 11.1, color=G, edgecolor="w")
bot = 0
for i in range(5):
    ax.bar(1, 6.4, bottom=bot, color=O, edgecolor="w"); bot += 6.4
ax.text(0, 11.1 + 0.6, "11.1", ha="center", fontsize=9); ax.text(1, 32.0 + 0.6, "32.0", ha="center", fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(["reliable clue", "five weak clues\n(6.4 each, stacked)"], fontsize=8)
ax.set_ylabel(r"sum of weights $\sum|w_i|$"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_il_weights.pdf")

# (c) geometry: same distance to the boundary, different box size
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.6))
d = 1.0
for ax, kind in zip(axs, "AB"):
    ax.set_aspect("equal"); ax.set_xlim(-1.6, 1.9); ax.set_ylim(-1.6, 1.6)
    ax.add_patch(Circle((0, 0), d, fill=False, ec=B_, lw=1.5))
    if kind == "A":
        ax.axvline(d, color="k", lw=2); eps = d
        ax.set_title("evidence in one input", fontsize=10)
    else:
        xx = np.linspace(-1.6, 1.9, 10); ax.plot(xx, np.sqrt(2) * d - xx, "k", lw=2); eps = d / np.sqrt(2)
        ax.set_title("same evidence split over two inputs", fontsize=10)
    ax.add_patch(Rectangle((-eps, -eps), 2 * eps, 2 * eps, fill=False, ec=R, lw=1.5, ls="--"))
    ax.plot(0, 0, "ko", ms=5); ax.text(0.04, 0.06, "a point", fontsize=8)
    ax.text(-1.5, -1.45, f"blue circle: total change needed = {d:.2f}\nred box: change per input needed = {eps:.2f}", fontsize=8, va="bottom")
    ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
fig.tight_layout(); fig.savefig("fig_il_geometry.pdf")

# (d) share flipped vs number of weak clues the attacker may touch (Model A, eps=0.3)
ks = list(range(0, 6)); vals = {k: [] for k in ks}
for s in f.SEEDS:
    (xt, yt), (xs_, ys_) = f.data(s); torch.manual_seed(s)
    A = f.train(f.MLP(6), xt, yt)
    for k in ks:
        mk = [0] + [1] * k + [0] * (5 - k)
        vals[k].append(f.fgsm(A, xs_, ys_, 0.3, mk) if k else 0.0)
m_ = [np.mean(vals[k]) for k in ks]; s_ = [np.std(vals[k], ddof=1) for k in ks]
fig, ax = plt.subplots(figsize=(5.4, 3.3))
ax.bar(ks, m_, yerr=s_, color=O, capsize=3)
ax.set_xlabel("number of weak clues the attacker may nudge (0.3 each)"); ax.set_ylabel("share of answers flipped")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_il_kdims.pdf"); print("kdims", np.round(m_, 3))

# (e) permutation importance bars
bayes = pickle.load(open("feature_reliance_unclipped_main_bayes_modelc.pkl", "rb"))
bd, md = bayes["bayes"], bayes["modelc"]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
w_ = 0.35
for j, (lab, col) in enumerate([("best possible rule", "#7f7f7f"), ("trained network", B_)]):
    keys = ("pb_rel", "pb_w") if j == 0 else ("pm_rel", "pm_w")
    ax.bar(np.arange(2) + (j - 0.5) * w_, [np.mean(bd[k]) for k in keys], w_,
           yerr=[np.std(bd[k], ddof=1) for k in keys], color=col, capsize=3, label=lab)
ax.set_xticks([0, 1]); ax.set_xticklabels(["reliable clue\nscrambled", "five weak clues\nscrambled"])
ax.set_ylabel("accuracy lost"); ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_il_perm.pdf")

# (f) Model C: frozen weak slots vs size of leftover weights
sc = [1, 3, 10, 100]
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.errorbar(range(4), [np.mean(md[s]) for s in sc], yerr=[np.std(md[s], ddof=1) for s in sc], fmt="o-", color=R, capsize=3)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{s}x" for s in sc]); ax.set_xlabel("size of the leftover weights (x ordinary)")
ax.set_ylabel("share flipped (weak slots nudged)"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_il_modelc.pdf")

# (g) frontier: accuracy vs weak-clue attack for every regulariser setting
regd = pickle.load(open("feature_reliance_unclipped_reg_phase.pkl", "rb"))["reg"]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for key, lab, col, mk in [("L1", "$L_1$", B_, "o"), ("GL", "group Lasso", G, "s"), ("EN", "elastic net", "#9467bd", "^"),
                          ("L2", "$L_2$ (ordinary shrinkage)", R, "x"), ("prune", "delete weak inputs", "#7f7f7f", "d")]:
    pts = [(np.mean(r["acc"]), np.mean(r["weak"])) for r in regd[key]]
    pts = [p for p in pts if p[0] > 0.6]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=col, label=lab, ms=5)
ax.set_xlabel("accuracy"); ax.set_ylabel("share flipped by an attack on the weak clues")
ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_il_frontier.pdf")

# (h) real-data summaries
rd = pickle.load(open("root_cause_realdata.pkl", "rb"))
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.1))
for ax, task, lab in [(axs[0], "3v7", "3 vs 7"), (axs[1], "10class", "all ten digits")]:
    v = rd[task]; ks_ = sorted(v[0])
    inf = np.array([[s[k]["inf"] for k in ks_] for s in v]); l2 = np.array([[s[k]["l2"] for k in ks_] for s in v])
    ax.plot(range(len(ks_)), inf.mean(0) / inf.mean(0)[0], "o-", color=R, label="per-pixel flip size")
    ax.plot(range(len(ks_)), l2.mean(0) / l2.mean(0)[0], "s-", color=B_, label="total-change flip size")
    ax.plot(range(len(ks_)), 1 / np.sqrt(np.array(ks_)), "k:", label=r"toy-world prediction ($1/\sqrt{k}$)")
    ax.set_xticks(range(len(ks_))); ax.set_xticklabels(ks_); ax.set_xlabel("how widely each clue is spread"); ax.set_title(lab, fontsize=9)
    ax.set_ylim(0, 1.15); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("flip size (relative to unmixed)"); axs[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("fig_il_mnist_mix.pdf")

ci = pickle.load(open("root_cause_cifar.pkl", "rb")); ms_ = sorted(ci[0])
acc_ = np.array([[s[m]["acc"] for m in ms_] for s in ci]); inf_ = np.array([[s[m]["inf"] for m in ms_] for s in ci])
l2_ = np.array([[s[m]["l2"] for m in ms_] for s in ci])
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.1))
axs[0].plot(range(5), acc_.mean(0), "o-", color=G); axs[0].set_ylabel("accuracy")
axs[1].plot(range(5), inf_.mean(0) / inf_.mean(0)[0], "o-", color=R, label="per-pixel flip size")
axs[1].plot(range(5), l2_.mean(0) / l2_.mean(0)[0], "s-", color=B_, label="total-change flip size")
axs[1].set_ylabel("flip size (relative to $m$=16)"); axs[1].legend(frameon=False, fontsize=8)
for a in axs:
    a.set_xticks(range(5)); a.set_xticklabels(ms_); a.set_xlabel("number of image patterns used ($m$)"); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_il_cifar_m.pdf")
