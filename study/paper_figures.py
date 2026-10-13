"""Individual figures for study/paper/root_cause_study.tex from study/results/toy_*.pkl"""
import pickle, numpy as np
import matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
R = "/home/akayyam/adv_reason/study/results/"; F = "/home/akayyam/adv_reason/study/paper/figs/"
L = lambda n: pickle.load(open(R + f"toy_{n}.pkl", "rb"))
B, O, G, K, RED = "#1f77b4", "#ff7f0e", "#2ca02c", "k", "#d62728"
def fin(ax): ax.spines[["top", "right"]].set_visible(False)

h = L("h10"); fig, ax = plt.subplots(figsize=(5.4, 3.5))
ax.plot(h["eps"], h["best_linear"], "s-", color=G, label="best achievable rule (the inevitable ceiling)")
ax.errorbar(h["eps"], h["mlp"], yerr=h["mlp_sd"], fmt="o-", color=B, label="trained network")
ax.plot(h["eps"], h["bayes"], "--", color=K, label="best clean rule (Bayes)")
ax.fill_between(h["eps"], h["mlp"], h["best_linear"], color=G, alpha=0.12)
ax.text(0.3, 0.6, "avoidable\nonly by changing\nthe objective", fontsize=8, ha="center", color="#2a6b2a")
ax.set_xlabel(r"allowed per-input change $\epsilon$ ($\ell_\infty$)"); ax.set_ylabel("accuracy under attack"); ax.legend(fontsize=7, frameon=False, loc="lower left"); fin(ax)
fig.tight_layout(); fig.savefig(F + "h10_ceiling.pdf")

h = L("h4a"); fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.plot(h["ss"], h["bayes"], "--", color=K, label="best rule"); ax.plot(h["ss"], h["mlp"], "o-", color=B, label="trained network")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("number of informative inputs (out of 64)"); ax.set_ylabel(r"median $\ell_\infty$ flip size"); ax.legend(fontsize=8, frameon=False); fin(ax)
fig.tight_layout(); fig.savefig(F + "h4a_sparse.pdf")

h = L("h8"); fig, ax = plt.subplots(figsize=(5.0, 3.4)); ks = np.array(h["ks"]); r = np.array(h["ratio"], float); m = ks >= 4
ax.plot(ks[m], r[m], "o-", color=RED, label="measured"); ax.plot(ks, h["pred_ratio"], ":", color=K, label=r"$\|w\|_1/(0.67\,\|w\|_2)$")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("number of inputs $k$ sharing the evidence"); ax.set_ylabel("random-noise size / worst-case size"); ax.legend(fontsize=8, frameon=False); fin(ax)
fig.tight_layout(); fig.savefig(F + "h8_noise.pdf")

h = L("h7"); fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3)); w = 0.27; names = ["oracle (best rule)", r"$\ell_2$ max-margin (what gradient descent reaches)", r"$\ell_\infty$ max-margin"]
for j, m_ in enumerate(h["methods"]):
    axs[0].bar(np.arange(len(h["ss"])) + (j - 1) * w, [np.mean(h["acc"][(s, m_)]) for s in h["ss"]], w, color=[K, O, G][j] if j != 0 else "#7f7f7f")
    axs[1].bar(np.arange(len(h["ss"])) + (j - 1) * w, [np.nanmean(h["rad"][(s, m_)]) for s in h["ss"]], w, color=[K, O, G][j] if j != 0 else "#7f7f7f", label=names[j])
for a, yl in zip(axs, ["test accuracy", r"median $\ell_\infty$ flip size"]):
    a.set_xticks(range(len(h["ss"]))); a.set_xticklabels(h["ss"]); a.set_xlabel("informative inputs (of 100); 60 training examples"); a.set_ylabel(yl); fin(a)
axs[0].set_ylim(0.5, 1.0); axs[1].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(F + "h7_margin.pdf")

h = L("h6"); fig, ax = plt.subplots(figsize=(5.0, 3.3))
ax.plot(h["ns"], [np.mean(h["inf"][n]) for n in h["ns"]], "o-", color=B, label="trained network"); ax.axhline(np.mean(h["bayes_inf"]), color=K, ls="--", label="best rule")
ax.set_xscale("log"); ax.set_ylim(0, 0.16); ax.set_xlabel("training examples"); ax.set_ylabel(r"median $\ell_\infty$ flip size"); ax.legend(fontsize=8, frameon=False); fin(ax)
fig.tight_layout(); fig.savefig(F + "h6_data.pdf")

fig, ax = plt.subplots(figsize=(5.4, 3.4))
for tag, lab, col in [("h5", "with weight decay", B), ("h5_wd0", "no weight decay", O)]:
    h = L(tag); so = h["sig_offs"]
    ax.plot(range(len(so)), [min(np.mean(h["normal"][s]), 8.0) for s in so], "o-", color=col, label=f"{lab}: off-manifold")
    ax.plot(range(len(so)), [np.mean(h["tangent"][s]) for s in so], "s:", color=col, label=f"{lab}: on-manifold")
ax.set_xticks(range(len(so))); ax.set_xticklabels(so); ax.set_xlabel("noise level off the manifold"); ax.set_ylabel(r"median $\ell_2$ flip size (capped at 8)"); ax.legend(fontsize=7, frameon=False); fin(ax)
fig.tight_layout(); fig.savefig(F + "h5_manifold.pdf")
print("ok")

# ---------------- H2b transfer test, H9 invariance, H3 shape/texture exp1
h = L("h2b"); fig, ax = plt.subplots(figsize=(5.2, 3.3)); es = h["eps"]
for i, e in enumerate(es):
    ax.scatter([i - 0.12] * 10, h["adv"][e], color=B, s=14, alpha=0.7, label="trained on adversarial changes only" if i == 0 else None)
    ax.scatter(i + 0.12 + 0 * np.arange(10), h["noise"][e], color="#7f7f7f", s=14, alpha=0.7, label="control: random noise of the same size" if i == 0 else None)
ax.axhline(0.5, color=K, lw=0.8, ls="--"); ax.set_xticks(range(len(es))); ax.set_xticklabels(es); ax.set_xlabel(r"size of the change $\epsilon$"); ax.set_ylabel("accuracy on clean data, true labels")
ax.set_ylim(-0.05, 1.25); ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=1); fin(ax); fig.tight_layout(); fig.savefig(F + "h2b_transfer.pdf")

h = pickle.load(open(R + "invariance.pkl", "rb")); amps = h["amps"]; fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2)); w = 0.26
for j, (k, lab, col) in enumerate([("mlp", "fully connected", O), ("cnn", "translation-invariant CNN", B), ("mf", "hand-built matched filter", G)]):
    axs[0].bar(np.arange(3) + (j - 1) * w, [h["rows"][a][k + "_acc"] for a in amps], w, color=col, label=lab)
    axs[1].bar(np.arange(3) + (j - 1) * w, [h["rows"][a][k + "_inf"] for a in amps], w, color=col)
for a, yl in zip(axs, ["accuracy", r"median $\ell_\infty$ flip size"]):
    a.set_xticks(range(3)); a.set_xticklabels(amps); a.set_xlabel("strength of the hidden pattern"); a.set_ylabel(yl); fin(a)
axs[0].set_ylim(0, 1.25); axs[0].legend(fontsize=7, frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(F + "h9_invariance.pdf")


from scipy.stats import norm as N01
st = pickle.load(open(R + "shape_texture.pkl", "rb")); e1 = st["exp1"]; e2 = st["exp2"]
names = {"low_spread": "smooth, spread", "low_local": "smooth, concentrated (shape-like)", "high_spread": "fine pattern, spread (texture-like)", "high_local": "fine pattern, concentrated"}
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.5))
sp = np.array([e1["spread"][c] for c in names]); xx = np.linspace(6, 17, 50)
axs[0].plot(xx, 1.72 / xx, "--", color=K, label=r"best rule: $1.72/(\ell_1/\ell_2)$")
for key, lab, col, mk in [("mlp", "fully connected network", O, "s"), ("cnn", "convolutional network", B, "o")]:
    axs[0].scatter(sp, [e1["rows"][c][key + "_inf"] for c in names], color=col, marker=mk, s=40, label=lab, zorder=3)
axs[0].set_xlabel(r"how widely the cue is spread ($\ell_1/\ell_2$ of its template)"); axs[0].set_ylabel(r"median $\ell_\infty$ flip size"); axs[0].legend(fontsize=7, frameon=False); fin(axs[0])
conds = e2["conds"]; w = 0.35
bay = [N01.cdf(0.5 * (a * a - b * b) / np.sqrt(a * a + b * b)) for a, b in conds]
axs[1].bar(np.arange(4) - w / 2, bay, w, color="#7f7f7f", label="best rule"); axs[1].bar(np.arange(4) + w / 2, [e2["rows"][c]["follow_shape"] for c in conds], w, color=B, label="convolutional network")
axs[1].set_xticks(range(4)); axs[1].set_xticklabels([f"shape {a}\ntexture {b}" for a, b in conds], fontsize=7); axs[1].set_ylabel("share labelled by the shape cue"); axs[1].axhline(0.5, color=K, lw=0.6, ls=":"); axs[1].legend(fontsize=7, frameon=False); fin(axs[1])
fig.tight_layout(); fig.savefig(F + "h3_shape_texture.pdf")
print({c: round(e2["rows"][c]["inf"], 3) for c in conds}, [round(b, 3) for b in bay])
