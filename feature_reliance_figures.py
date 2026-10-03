"""Figures for feature_reliance_paper.tex from feature_reliance_unclipped_*.pkl."""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType so figure text is searchable
import matplotlib.pyplot as plt

main = pickle.load(open("feature_reliance_unclipped_main_bayes_modelc.pkl", "rb"))["main"]
ph = pickle.load(open("feature_reliance_unclipped_reg_phase.pkl", "rb"))["phase"]
EPS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

fig, ax = plt.subplots(figsize=(5.2, 3.6))
series = [(("A", "all"), "A: all dims", "#1f77b4", "-"), (("A", "reliable"), "A: reliable only", "#2ca02c", "-"),
          (("A", "weak5"), "A: five weak dims", "#ff7f0e", "-"), (("A", "weak1"), "A: one weak dim", "#d62728", "-"),
          (("B", "reliable"), "B: reliable only", "#1f77b4", "--")]
for (k, lab, c, ls) in series:
    v = np.array([main[(k[0], k[1], e)] for e in EPS])
    m, s = v.mean(1), v.std(1, ddof=1)
    ax.plot(EPS, m, ls, color=c, label=lab)
    ax.fill_between(EPS, m - s, m + s, color=c, alpha=0.15)
ax.set_xlabel(r"$\ell_\infty$ budget $\epsilon$"); ax.set_ylabel("FGSM success")
ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_fr_curves.pdf")

S0, AL = [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.4, 0.6]
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
for a, key, title, fmt in [(axs[0], "ratio", "signal:noise weight ratio", "{:.1f}"),
                           (axs[1], "weak", r"weak-only attack success ($\epsilon$=0.2)", "{:.2f}"),
                           (axs[2], "acc", "clean accuracy", "{:.2f}")]:
    M = np.array([[np.mean(ph[(s, al, 1e-2)][key]) for al in AL] for s in S0])
    im = a.imshow(M, origin="lower", cmap="viridis")
    a.set_xticks(range(4)); a.set_xticklabels(AL); a.set_yticks(range(5)); a.set_yticklabels(S0)
    a.set_xlabel(r"$\alpha$ (weak-feature strength)"); a.set_ylabel(r"$\sigma_0$ (reliable-dim noise)")
    a.set_title(title, fontsize=9)
    for i in range(5):
        for j in range(4):
            a.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7, color="w")
fig.tight_layout(); fig.savefig("fig_fr_phase.pdf")
