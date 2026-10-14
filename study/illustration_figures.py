"""Illustrative (picture-style) figures for study/paper/root_cause_study.tex. Output: study/paper/figs/ill_*.pdf"""
import sys, math, pickle
import numpy as np, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "/home/akayyam/adv_reason"); sys.path.insert(0, "/home/akayyam/adv_reason/study")
import real_checks as rcx
import shape_texture as st
import invariance as inv
from scipy.optimize import linprog
from sklearn.svm import SVC
F = "/home/akayyam/adv_reason/study/paper/figs/"; DEV = rcx.DEV
B, O, G, K, RED, PU = "#1f77b4", "#ff7f0e", "#2ca02c", "k", "#d62728", "#9467bd"
def fin(ax): ax.spines[["top", "right"]].set_visible(False)
which = sys.argv[1:] or ["a", "c", "d", "e", "f", "g", "i", "j"]


# ---------------- A: inevitable vs avoidable (2-D schematic)
def fig_a():
    rng = np.random.default_rng(3); n = 120
    y = rng.integers(0, 2, n); s = (y - 0.5)
    x = s * 1.0 + 0.3 * rng.standard_normal(n); z = s * 0.4 + 0.25 * rng.standard_normal(n)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2)); eps = 0.3
    # left: inevitable -- classes whose eps-squares overlap
    ax = axs[0]; rng2 = np.random.default_rng(1); m = 14; yy = rng2.integers(0, 2, m); xx = (yy - 0.5) * 0.35 + 0.12 * rng2.standard_normal(m); zz = rng2.uniform(-0.8, 0.8, m)
    for xi, zi, yi in zip(xx, zz, yy):
        ax.add_patch(Rectangle((xi - 0.2, zi - 0.2), 0.4, 0.4, fill=True, alpha=0.12, color=[B, O][yi], lw=0))
    ax.scatter(xx, zz, c=[[B, O][v] for v in yy], s=18, zorder=3); ax.set_xlim(-0.9, 0.9); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
    ax.set_title("inevitable: classes closer than the allowed change\n(no classifier can be right and safe)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    # right: avoidable -- tilted clean boundary vs vertical robust boundary
    ax = axs[1]; ax.scatter(x, z, c=[[B, O][v] for v in y], s=14, alpha=0.8)
    xs = np.linspace(-1.1, 1.1, 10); ax.plot(xs, -11.1 * xs / 6.4, "k--", label="most accurate rule (uses both clues)"); ax.axvline(0, color=G, lw=2, label="most robust rule (reliable clue only)")
    p = (0.35, 0.1); ax.add_patch(Rectangle((p[0] - eps, p[1] - eps), 2 * eps, 2 * eps, fill=False, ec=RED, lw=1.5)); ax.scatter(*p, color=O, s=60, edgecolor="k", zorder=4)
    ax.text(0.42, 0.42, "allowed\nchange", color=RED, fontsize=8); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.0, 1.0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("avoidable: a safer rule exists;\nthe accuracy objective does not pick it", fontsize=9); ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(F + "ill_a_split.pdf")


# ---------------- C: l2 vs l_inf max-margin (2-D)
def fig_c():
    best = None
    for seed in range(200):
        rng = np.random.default_rng(seed); n = 10; y = np.array([-1, 1] * 5); x = np.c_[y * 0.9 + 0.35 * rng.standard_normal(n), 1.1 * rng.standard_normal(n)]
        svm = SVC(kernel="linear", C=1e6).fit(x, y); w2 = svm.coef_[0]; b2 = svm.intercept_[0]
        A = -(y[:, None] * np.hstack([x, -x])); lp = linprog(np.ones(4), A_ub=A, b_ub=-np.ones(n), bounds=(0, None), method="highs")
        if not lp.success: continue
        wl = lp.x[:2] - lp.x[2:]
        if abs(w2[0]) > 1e-6 and (np.sign(svm.predict(x)) == y).all():
            ang = abs(math.degrees(math.atan2(w2[1], w2[0]))); angl = abs(math.degrees(math.atan2(wl[1], wl[0])))
            sc = ang - angl                                  # SVM tilted, LP close to using the informative input only
            if best is None or sc > best[0]: best = (sc, seed, x, y, w2, b2, wl)
    _, seed, x, y, w2, b2, wl = best
    fig, ax = plt.subplots(figsize=(5.4, 4.2)); ax.scatter(x[:, 0], x[:, 1], c=[[B, O][int(v > 0)] for v in y], s=40, zorder=3)
    xs = np.linspace(-2, 2, 10)
    ax.plot(xs, -(w2[0] * xs + b2) / w2[1], color=RED, lw=2, label=r"largest $\ell_2$ margin (what gradient descent finds)")
    ax.plot(xs, -(wl[0] * xs) / wl[1] if abs(wl[1]) > 1e-9 else np.zeros_like(xs), color=G, lw=2, label=r"largest $\ell_\infty$ margin")
    if abs(wl[1]) <= 1e-9: ax.axvline(0, color=G, lw=2)
    ax.set_xlim(-2, 2); ax.set_ylim(-2.4, 2.4); ax.set_xlabel("informative input"); ax.set_ylabel("irrelevant input"); ax.legend(fontsize=7, frameon=False, loc="upper left"); fin(ax)
    fig.tight_layout(); fig.savefig(F + "ill_c_margin.pdf")


# ---------------- D: shape/texture cues
def fig_d():
    fig, axs = plt.subplots(2, 5, figsize=(10, 4.4)); g = torch.Generator().manual_seed(2)
    names = [("low_spread", "smooth, spread"), ("low_local", "smooth, concentrated\n(shape-like)"), ("high_spread", "fine, spread\n(texture-like)"), ("high_local", "fine, concentrated")]
    for j, (k, t) in enumerate(names):
        axs[0, j].imshow(st.T[k].reshape(16, 16), cmap="RdBu_r"); axs[0, j].set_title(t, fontsize=8); axs[0, j].axis("off")
        x, _ = st.make(1, {k: 3.3}, g); y1 = torch.tensor([1]); xi = (0.5 * 3.3 * torch.tensor(st.T[k]).float() + st.SIGMA * torch.randn(256, generator=g))
        axs[1, j].imshow(xi.view(16, 16), cmap="gray"); axs[1, j].axis("off")
    axs[0, 4].axis("off"); axs[0, 4].text(0.0, 0.5, "cue pattern\n(added with a\nclass-dependent sign)", fontsize=8, va="center")
    xc = st.SIGMA * torch.randn(256, generator=g) + 0.5 * 3.3 * torch.tensor(st.T["low_local"]).float() - 0.5 * 3.3 * torch.tensor(st.T["high_spread"]).float()
    axs[1, 4].imshow(xc.view(16, 16), cmap="gray"); axs[1, 4].set_title("cue conflict: shape says 1,\ntexture says 0", fontsize=8); axs[1, 4].axis("off")
    axs[1, 0].text(-0.05, 0.5, "noisy image\n(label 1)", transform=axs[1, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(F + "ill_d_cues.pdf")


# ---------------- E: invariance task and what each model learns
def fig_e():
    g = torch.Generator().manual_seed(5); amp = 5.0
    xt, yt = inv.make(3000, amp, g)
    torch.manual_seed(0); cnn = inv.train(inv.CNNg(), xt, yt); torch.manual_seed(0); mlp = inv.train(inv.MLP(), xt, yt)
    fig, axs = plt.subplots(2, 6, figsize=(11, 4.0))
    xs, ys = inv.make(40, amp, torch.Generator().manual_seed(9)); idx = [int(i) for i in torch.where(ys == 1)[0][:2]] + [int(i) for i in torch.where(ys == 0)[0][:1]]
    for j, i in enumerate(idx):
        axs[0, j].imshow(xs[i].view(16, 16), cmap="gray"); axs[0, j].set_title("pattern present (box)" if ys[i] == 1 else "no pattern", fontsize=8); axs[0, j].axis("off")
        if ys[i] == 1:
            sc_ = nn.functional.conv2d(xs[i].view(1, 1, 16, 16), inv.TMPL)[0, 0]; r_, c_ = divmod(int(sc_.argmax()), sc_.shape[1])
            axs[0, j].add_patch(Rectangle((c_ - 0.5, r_ - 0.5), 3, 3, fill=False, ec=RED, lw=1.2))
    axs[0, 3].imshow(inv.TMPL[0, 0], cmap="gray", vmin=0, vmax=0.5); axs[0, 3].set_title("the pattern\n(3x3 block)", fontsize=8); axs[0, 3].axis("off")
    W = cnn.c.weight.detach()[:, 0]
    for j in range(2): axs[0, 4 + j].imshow(W[j], cmap="RdBu_r"); axs[0, 4 + j].set_title(f"CNN filter {j + 1}", fontsize=8); axs[0, 4 + j].axis("off")
    Wm = mlp.n[0].weight.detach().view(-1, 16, 16)
    for j in range(6): axs[1, j].imshow(Wm[j], cmap="RdBu_r"); axs[1, j].set_title(f"MLP hidden unit {j + 1}", fontsize=8); axs[1, j].axis("off")
    fig.tight_layout(); fig.savefig(F + "ill_e_invariance.pdf")


# ---------------- F: gradient maps with appended irrelevant inputs (MNIST)
def fig_f():
    xtr, ytr, xte, yte = rcx.load("mnist"); x, y = rcx.subset(xtr, ytr, 10000, 0); gg = torch.Generator().manual_seed(100)
    xn = torch.cat([x, 0.3 * torch.randn(len(x), 784, generator=gg)], 1).to(DEV); xt_ = torch.cat([xte, 0.3 * torch.randn(len(xte), 784, generator=gg)], 1)[:500].to(DEV)
    torch.manual_seed(0); m = nn.Sequential(nn.Linear(1568, 256), nn.ReLU(), nn.Linear(256, 10)).to(DEV); yt_ = y.to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    for _ in range(20):
        p = torch.randperm(len(xn), device=DEV)
        for i in range(0, len(xn), 128):
            b = p[i:i + 128]; opt.zero_grad(); nn.functional.cross_entropy(m(xn[b]), yt_[b]).backward(); opt.step()
    m.eval(); xq = xt_.clone().requires_grad_(True); nn.functional.cross_entropy(m(xq), yte[:500].to(DEV), reduction="sum").backward()
    gmap = xq.grad.abs().mean(0).cpu(); real, noise = gmap[:784].view(28, 28), gmap[784:].view(28, 28); vmax = float(max(real.max(), noise.max()))
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    axs[0].imshow(real, cmap="magma", vmin=0, vmax=vmax); axs[0].set_title("sensitivity to the 784 real pixels", fontsize=9); axs[0].axis("off")
    axs[1].imshow(noise, cmap="magma", vmin=0, vmax=vmax); axs[1].set_title("sensitivity to the 784 irrelevant inputs\n(pure noise, same colour scale)", fontsize=9); axs[1].axis("off")
    axs[2].bar([0, 1], [float(real.sum()), float(noise.sum())], color=[B, O]); axs[2].set_xticks([0, 1]); axs[2].set_xticklabels(["real", "irrelevant"]); axs[2].set_ylabel("total |gradient|"); fin(axs[2])
    fig.tight_layout(); fig.savefig(F + "ill_f_irrelevant.pdf"); print("fig_f totals", float(real.sum()), float(noise.sum()))


# ---------------- G: where does the change go (principal components, frequency bands)
def l2_adv(model, x, y, eps_frac=1.3, hi=20.0):
    fs = rcx.flip_size(model, x, y, "2", hi, n=len(x)); return fs
def fig_g():
    rows = []
    for name, k, h, ch in [("mnist", 50, 28, 1), ("cifar", 200, 32, 3)]:
        xtr, ytr, xte, yte = rcx.load(name); x, y = rcx.subset(xtr, ytr, 10000, 0); m = rcx.train_model(name, x, y, 0)
        P = rcx.pca_projector(x, k); xs, ys = xte[:300].to(DEV), yte[:300].to(DEV)
        with torch.no_grad(): ok = (m(xs).argmax(1) == ys)
        i = int(torch.where(ok)[0][3]); xi, yi = xs[i:i + 1], ys[i:i + 1]
        fs = float(rcx.flip_size(m, xi, yi, "2", rcx.HI[name][1], n=1)[0]); eps = fs * 1.3
        delta = torch.zeros_like(xi)
        for _ in range(40):
            d = delta.clone().requires_grad_(True); g_ = torch.autograd.grad(nn.functional.cross_entropy(m(xi + d), yi), d)[0]
            delta = (delta + eps / 8 * g_ / g_.norm().clamp_min(1e-12)); delta = delta * min(1.0, eps / float(delta.norm()))
        delta = delta.detach(); top = delta @ P; rest = delta - top
        rows.append((name, h, ch, xi.cpu(), delta.cpu(), top.cpu(), rest.cpu(), float(top.norm() ** 2 / delta.norm() ** 2), float(m(xi + delta).argmax()), int(yi)))
    fig, axs = plt.subplots(2, 5, figsize=(11, 4.6))
    def show(ax, v, h, ch, sym=True, scale=None):
        im = v.view(ch, h, h).permute(1, 2, 0).numpy() if ch == 3 else v.view(h, h).numpy()
        if sym:
            a = scale or np.abs(im).max(); ax.imshow(np.clip(im / (2 * a) + 0.5, 0, 1) if ch == 3 else im, cmap="RdBu_r", vmin=-a, vmax=a)
        else: ax.imshow(np.clip(im + 0.5, 0, 1), cmap="gray" if ch == 1 else None)
        ax.axis("off")
    for r, (name, h, ch, xi, delta, top, rest, frac, pred, yi) in enumerate(rows):
        sc = float(delta.abs().max())
        show(axs[r, 0], xi, h, ch, sym=False); axs[r, 0].set_title(f"original (class {yi})" if r == 0 else "original", fontsize=8)
        show(axs[r, 1], delta, h, ch, scale=sc); axs[r, 1].set_title("smallest change that flips the answer", fontsize=8)
        show(axs[r, 2], top, h, ch, scale=sc); axs[r, 2].set_title(f"part along the top {50 if name == 'mnist' else 200} components\n({100 * frac:.0f}% of its energy)", fontsize=8)
        show(axs[r, 3], rest, h, ch, scale=sc); axs[r, 3].set_title(f"the rest ({100 * (1 - frac):.0f}%)", fontsize=8)
        show(axs[r, 4], xi + delta, h, ch, sym=False); axs[r, 4].set_title(f"changed (answer {pred:.0f})", fontsize=8)
        axs[r, 0].text(-0.08, 0.5, "MNIST" if name == "mnist" else "CIFAR-10", transform=axs[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.tight_layout(); fig.savefig(F + "ill_g_where.pdf"); print("fig_g energy fractions", [round(r[7], 2) for r in rows])


# ---------------- I: how the transfer-test data is built
def fig_i():
    xtr, ytr, xte, yte = rcx.load("mnist"); x, y = rcx.subset(xtr, ytr, 10000, 0); m = rcx.train_model("mnist", x, y, 0)
    xs = xte[:200].to(DEV); ys = yte[:200].to(DEV); g = torch.Generator().manual_seed(4); t = torch.randint(0, 10, (200,), generator=g).to(DEV)
    t = torch.where(t == ys, (t + 1) % 10, t); xa = rcx.targeted_pgd(m, xs, t, 0.3).detach()
    with torch.no_grad(): pa = m(xa).argmax(1)
    idx = [i for i in range(200) if pa[i] == t[i]][:6]
    fig, axs = plt.subplots(3, 6, figsize=(8.6, 4.6))
    for j, i in enumerate(idx):
        axs[0, j].imshow((xs[i] + 0.5).view(28, 28).cpu(), cmap="gray"); axs[0, j].set_title(f"clean, true label {int(ys[i])}", fontsize=7)
        axs[1, j].imshow((xa[i] - xs[i]).view(28, 28).cpu(), cmap="RdBu_r", vmin=-0.3, vmax=0.3); axs[1, j].set_title(f"change toward {int(t[i])}", fontsize=7)
        axs[2, j].imshow((xa[i] + 0.5).view(28, 28).cpu(), cmap="gray", vmin=0, vmax=1); axs[2, j].set_title(f"training example,\nlabel {int(t[i])}", fontsize=7)
    for a in axs.ravel(): a.axis("off")
    fig.tight_layout(); fig.savefig(F + "ill_i_transfer.pdf")


# ---------------- J: graphical scorecard
def fig_j():
    rows = [("H1  evidence spread over many inputs", "inevitable", 2, 1, "geometry holds; MNIST already spread"),
            ("H2  small, not unreliable; features carry signal", "inevitable", 2, 2, "transfer works on MNIST, weaker CIFAR-10"),
            ("H4a  irrelevant inputs ignored badly", "avoidable", 2, 2, "about 2x in toy, MNIST, wide MLP"),
            ("H4b  weight sparsity as a fix", "avoidable", 2, 0, ""),
            ("H4c  activity sparsity", "none", 3, 0, ""),
            ("H8  noise vs worst case", "inevitable", 2, 1, "right order of magnitude"),
            ("H10  objective (adversarial training)", "avoidable", 2, 2, "closes the toy gap; costs accuracy"),
            ("H7  optimiser bias (l2 vs l_inf margin)", "avoidable", 2, 2, "+50% flip size on MNIST, F-MNIST"),
            ("H6  data quantity", "none/avoidable", 3, 3, "none (standard); strong (adversarial)"),
            ("H5  off-manifold directions", "avoidable", 1, 1, "CIFAR-10 only, with a caveat"),
            ("H3  shape vs texture / frequency", "inevitable + arch.", 1, 3, "spread, not frequency, matters"),
            ("H9  architecture / invariance", "avoidable", 2, 1, "helps MNIST, not consistently")]
    col = {0: "#eeeeee", 1: "#f7d774", 2: "#8fd18f", 3: "#e8a0a0"}; lab = {0: "not tested", 1: "partly", 2: "supported", 3: "not supported"}
    fig, ax = plt.subplots(figsize=(10.5, 5.4)); ax.set_xlim(0, 10); ax.set_ylim(0, len(rows) + 1.2); ax.axis("off")
    for j, h in enumerate(["hypothesis", "kind", "toy world", "real data", "note"]): ax.text([0.05, 4.4, 5.6, 6.9, 8.1][j], len(rows) + 0.55, h, fontsize=9, weight="bold")
    for i, (name, kind, toy, real, note) in enumerate(rows):
        yy = len(rows) - i - 0.2
        ax.text(0.05, yy + 0.3, name, fontsize=8, va="center"); ax.text(4.4, yy + 0.3, kind, fontsize=8, va="center")
        for x0, v in [(5.6, toy), (6.9, real)]:
            ax.add_patch(Rectangle((x0, yy), 1.2, 0.6, color=col[v])); ax.text(x0 + 0.6, yy + 0.3, lab[v], ha="center", va="center", fontsize=7)
        ax.text(8.15, yy + 0.3, note, fontsize=6.5, va="center")
    fig.tight_layout(); fig.savefig(F + "ill_j_scorecard.pdf")


for w in which: globals()["fig_" + w](); print("done", w, flush=True)
