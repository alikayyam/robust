import pickle, numpy as np
import matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
R = "/home/akayyam/adv_reason/study/results/"
L = lambda n: pickle.load(open(R + f"toy_{n}.pkl", "rb"))
fig, ax = plt.subplots(2, 3, figsize=(13, 7.2))
# H10
h = L("h10"); a = ax[0, 0]
a.plot(h["eps"], h["bayes"], "k--", label="best clean rule (Bayes)"); a.errorbar(h["eps"], h["mlp"], yerr=h["mlp_sd"], fmt="o-", color="#1f77b4", label="trained network")
a.plot(h["eps"], h["best_linear"], "s-", color="#2ca02c", label="best achievable (linear rule)"); a.plot(h["eps"], h["reliable_only"], ":", color="#7f7f7f", label="reliable clue only")
a.set_xlabel(r"$\ell_\infty$ budget $\epsilon$"); a.set_ylabel("accuracy under attack"); a.set_title("H10: inevitable ceiling vs standard rule", fontsize=9); a.legend(fontsize=7, frameon=False)
# H4a
h = L("h4a"); a = ax[0, 1]
a.plot(h["ss"], h["bayes"], "k--", label="best rule"); a.plot(h["ss"], h["mlp"], "o-", color="#1f77b4", label="trained network")
a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("number of informative inputs (of 64)"); a.set_ylabel(r"median $\ell_\infty$ flip size"); a.set_title("H4a: sparse vs dense signal", fontsize=9); a.legend(fontsize=7, frameon=False)
# H8
h = L("h8"); a = ax[0, 2]; ks = np.array(h["ks"]); r = np.array(h["ratio"], float)
a.plot(ks, r, "o-", color="#d62728", label="measured: random-noise size / worst-case size"); a.plot(ks, h["pred_ratio"], "k:", label=r"$\|w\|_1/(0.67\|w\|_2)$"); a.set_xscale("log", base=2); a.set_yscale("log")
a.set_xlabel("number of inputs k"); a.set_title("H8: random noise vs worst case", fontsize=9); a.legend(fontsize=7, frameon=False)
# H7
h = L("h7"); a = ax[1, 0]; w = 0.27
for j, m in enumerate(h["methods"]):
    a.bar(np.arange(len(h["ss"])) + (j - 1) * w, [np.nanmean(h["rad"][(s, m)]) for s in h["ss"]], w, label=m)
a.set_xticks(range(len(h["ss"]))); a.set_xticklabels(h["ss"]); a.set_xlabel("number of informative inputs (of 100), n=60 samples"); a.set_ylabel(r"median $\ell_\infty$ flip size"); a.set_title("H7: which max-margin solution", fontsize=9); a.legend(fontsize=7, frameon=False)
# H6
h = L("h6"); a = ax[1, 1]
a.plot(h["ns"], [np.mean(h["inf"][n]) for n in h["ns"]], "o-", color="#1f77b4", label="trained network"); a.axhline(np.mean(h["bayes_inf"]), color="k", ls="--", label="best rule")
a.set_xscale("log"); a.set_ylim(0, 0.16); a.set_xlabel("training examples n"); a.set_ylabel(r"median $\ell_\infty$ flip size"); a.set_title("H6: data quantity (standard training)", fontsize=9); a.legend(fontsize=7, frameon=False)
# H5
a = ax[1, 2]
for tag, lab, col in [("h5", "weight decay", "#1f77b4"), ("h5_wd0", "no weight decay", "#ff7f0e")]:
    h = L(tag); so = h["sig_offs"]
    nm = [np.mean(h["normal"][s]) for s in so]; nm = [min(x, 8.0) for x in nm]
    a.plot(range(len(so)), nm, "o-", color=col, label=f"{lab}: off-manifold")
    a.plot(range(len(so)), [np.mean(h["tangent"][s]) for s in so], "s:", color=col, label=f"{lab}: on-manifold")
a.set_xticks(range(len(so))); a.set_xticklabels(so); a.set_xlabel("noise off the manifold"); a.set_ylabel(r"median $\ell_2$ flip size (capped at 8)"); a.set_title("H5: off-manifold directions", fontsize=9); a.legend(fontsize=6, frameon=False)
for x in ax.ravel(): x.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(R + "toy_overview.pdf"); fig.savefig(R + "toy_overview.png", dpi=110)
