"""Figures and summary numbers for root_cause_paper.tex, from root_cause_results.pkl."""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pickle.load(open("root_cause_results.pkl", "rb"))
E1, MAIN = R["e1"], R["main"]
NS = len(MAIN)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dc"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})


def ms(vals):
    v = np.asarray(vals, dtype=float)
    return v.mean(), v.std(ddof=1)


# ------------------------------------------------------------ Figure 1 (E1)
ks = sorted(E1[0].keys())
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
w1 = [E1[0][k]["w1"] for k in ks]
w2 = [E1[0][k]["w2"] for k in ks]
ax[0].loglog(ks, w1, "-o", color=ORANGE, lw=1.6, ms=4, label=r"$\|w\|_1$")
ax[0].loglog(ks, w2, "-s", color=VIOLET, lw=1.6, ms=4, label=r"$\|w\|_2$")
ax[0].set_xlabel("number of dimensions $k$")
ax[0].set_ylabel("Bayes weight norm")
ax[0].legend(loc="center left", fontsize=8)
ax[0].set_title("(a) same evidence, more dimensions", fontsize=9, loc="left")

for norm, col, mk, lab in [("inf", ORANGE, "o", r"$\ell_\infty$"), ("2", VIOLET, "s", r"$\ell_2$")]:
    b = [np.mean([s[k][f"bayes_{norm}"] for s in E1]) for k in ks]
    m = [ms([s[k][f"mlp_{norm}"] for s in E1]) for k in ks]
    ax[1].loglog(ks, b, "-", color=col, lw=1.6, label=f"Bayes, {lab} (exact)")
    ax[1].errorbar(ks, [x[0] for x in m], yerr=[x[1] for x in m], fmt=mk, color=col,
                   mfc="white", ms=5, lw=1, capsize=2, label=f"trained MLP, {lab}")
ref = E1[0][1]["bayes_inf"] / np.sqrt(np.array(ks, dtype=float))
ax[1].loglog(ks, ref, ":", color=MUTED, lw=1.2)
ax[1].text(ks[-2], ref[-2] * 0.62, r"$\propto 1/\sqrt{k}$", color=MUTED, fontsize=8, ha="center")
ax[1].set_xlabel("number of dimensions $k$")
ax[1].set_ylabel("median minimal flip radius")
ax[1].set_ylim(0.02, 0.8)
ax[1].legend(loc="lower left", fontsize=7, ncol=2, columnspacing=0.8, handlelength=1.5)
ax[1].set_title("(b) robustness vs. $k$ (accuracy fixed at 95%)", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig("fig_rc_dimensions.pdf")
plt.close(fig)

# ------------------------------------------------------------ Figure 2 (ECDF)
def pooled(key_fn):
    return np.sort(np.concatenate([key_fn(s) for s in MAIN]))


def ecdf(ax, a, **kw):
    a = a[np.isfinite(a)]
    y = np.arange(1, len(a) + 1) / len(a)
    ax.plot(np.concatenate([[0], a]), np.concatenate([[0], y]), **kw)


fig, ax = plt.subplots(figsize=(4.8, 3.1))
ecdf(ax, pooled(lambda s: s["bayes_inf"]), color=INK, lw=1.6, ls="--")
ecdf(ax, pooled(lambda s: s["A_inf"]), color=BLUE, lw=1.6)
ecdf(ax, pooled(lambda s: s["E4"]["B: signal only"]["inf"]), color=AQUA, lw=1.6)
ecdf(ax, pooled(lambda s: s["E4"]["W: weak only"]["inf"]), color=ORANGE, lw=1.6)
ax.set_xlim(0, 1.0)
ax.set_xlabel(r"$\ell_\infty$ budget $\epsilon$")
ax.set_ylabel("fraction of points flipped")
ax.text(0.28, 0.93, "weak dims only", color=ORANGE, fontsize=8, ha="right", va="center")
ax.text(0.46, 0.78, "all 6 dims", color=BLUE, fontsize=8, ha="left", va="center")
ax.text(0.80, 0.30, "signal only", color=AQUA, fontsize=8, ha="left", va="center")
fig.tight_layout()
fig.savefig("fig_rc_ecdf.pdf")
plt.close(fig)

# ------------------------------------------------------------ Figure 3 (E3)
names = ["signal (1 dim)", "one weak dim", "all 5 weak dims", "all 6 dims"]
fig, ax = plt.subplots(figsize=(4.8, 2.8))
xpos = np.arange(len(names))
for j, (key, col, lab) in enumerate([("bayes", INK, "Bayes-optimal (exact)"), ("A", BLUE, "trained MLP")]):
    vals = [ms([np.median(s["E3"][n][key]) for s in MAIN]) for n in names]
    ax.bar(xpos + (j - 0.5) * 0.36, [v[0] for v in vals], 0.34, yerr=[v[1] for v in vals],
           color=col, alpha=0.9, capsize=2, label=lab, error_kw=dict(lw=0.8))
ax.set_xticks(xpos)
ax.set_xticklabels(["signal\n(1 dim)", "one weak\ndim", "5 weak\ndims", "all 6\ndims"])
ax.set_ylabel(r"median $\ell_\infty$ flip radius" + "\n(larger = harder to attack)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig("fig_rc_subsets.pdf")
plt.close(fig)

# ------------------------------------------------------------ Figure 4 (E4)
fig, ax = plt.subplots(figsize=(4.8, 3.1))
def pt(name, key="inf"):
    return (ms([s["E4"][name]["acc"] for s in MAIN]),
            ms([np.median(s["E4"][name][key]) for s in MAIN]))
lam = sorted(MAIN[0]["L1"].keys())
accs = [ms([s["L1"][l]["acc"] for s in MAIN])[0] for l in lam]
infs = [ms([s["L1"][l]["inf"] for s in MAIN])[0] for l in lam]
ax.plot(accs, infs, "-o", color=BLUE, lw=1.4, ms=4)
ax.text(accs[2] + 0.002, infs[2] + 0.015, "trained MLP,\nincreasing $L_1$", color=BLUE, fontsize=7.5, ha="left")
for name, col, mk, lbl, dx, dy in [
    ("B: signal only", AQUA, "D", "signal only", 0.0, 0.025),
    ("W: weak only", ORANGE, "s", "weak only", 0.0, 0.025),
]:
    (a, _), (e, _) = pt(name)
    ax.plot([a], [e], mk, color=col, ms=7)
    ax.text(a + 0.001, e + dy, lbl, color=col, fontsize=8, ha="left")
for name, mk in [("Bayes: all", "*")]:
    (a, _), (e, _) = pt(name)
    ax.plot([a], [e], mk, color=INK, ms=9)
ax.text(pt("Bayes: all")[0][0] - 0.002, pt("Bayes: all")[1][0] - 0.03, "Bayes (all 6)", fontsize=7.5, ha="right")
ax.set_xlabel("clean test accuracy")
ax.set_ylim(0.17, 0.58)
ax.set_ylabel(r"median $\ell_\infty$ flip radius")
fig.tight_layout()
fig.savefig("fig_rc_tradeoff.pdf")
plt.close(fig)

# ------------------------------------------------------------ numbers
def f(x, d=3):
    return f"{x[0]:.{d}f}$\\pm${x[1]:.{d}f}"

print("== validation: closed form vs PGD, max abs err (over seeds):",
      max(s["closed_form_vs_pgd_maxabs"] for s in MAIN))
print("== Bayes acc closed form / trained A acc")
print(" bayes", f(ms([s["bayes_acc"] for s in MAIN]), 4), " A", f(ms([s["acc_A"] for s in MAIN]), 4))
print("== E1 table: k, ||w||1, ||w||2, bayes inf, mlp inf, bayes 2, mlp 2, mlp acc")
for k in ks:
    print(k, f"{E1[0][k]['w1']:.1f}", f"{E1[0][k]['w2']:.1f}",
          f"{np.mean([s[k]['bayes_inf'] for s in E1]):.3f}", f(ms([s[k]['mlp_inf'] for s in E1])),
          f"{np.mean([s[k]['bayes_2'] for s in E1]):.3f}", f(ms([s[k]['mlp_2'] for s in E1])),
          f(ms([s[k]['mlp_acc'] for s in E1])), f(ms([s[k]['bayes_acc'] for s in E1])))
# slope of log median eps_inf vs log k
lk = np.log(ks)
for key in ("bayes_inf", "mlp_inf", "bayes_2", "mlp_2"):
    y = np.log([np.mean([s[k][key] for s in E1]) for k in ks])
    print("slope", key, np.polyfit(lk, y, 1)[0])
print("== E2: attack success at eps for pooled points (bayes all / MLP A)")
for e in [0.1, 0.2, 0.3, 0.5]:
    b = np.concatenate([s["bayes_inf"] for s in MAIN]); a = np.concatenate([s["A_inf"] for s in MAIN])
    print(e, f"bayes {np.mean(b<=e):.3f}  A {np.mean(a<=e):.3f}")
print("median eps_inf bayes / A:", f(ms([np.median(s["bayes_inf"]) for s in MAIN])),
      f(ms([np.median(s["A_inf"]) for s in MAIN])))
print("median eps_2   bayes / A:", f(ms([np.median(s["bayes_2"]) for s in MAIN])),
      f(ms([np.median(s["A_2"]) for s in MAIN])))
print("== E3 (median eps_inf, bayes / A; success at eps=0.3 bayes / A)")
for n in names:
    bm = ms([np.median(s["E3"][n]["bayes"]) for s in MAIN]); am = ms([np.median(s["E3"][n]["A"]) for s in MAIN])
    b3 = np.mean(np.concatenate([s["E3"][n]["bayes"] for s in MAIN]) <= 0.3)
    a3 = np.mean(np.concatenate([s["E3"][n]["A"] for s in MAIN]) <= 0.3)
    print(n, f(bm), f(am), f"{b3:.3f} {a3:.3f}")
print("== E3 weights (Bayes):", np.round(MAIN[0]["w"], 2), " MLP col norms mean:",
      np.round(np.mean([s["colnorm_A"] for s in MAIN], 0), 2))
print("== E4: acc, median eps_inf, median eps_2")
for n in MAIN[0]["E4"]:
    print(n, f(ms([s["E4"][n]["acc"] for s in MAIN])),
          f(ms([np.median(s["E4"][n]["inf"]) for s in MAIN])),
          f(ms([np.median(s["E4"][n]["l2"]) for s in MAIN])))
print("== L1 sweep: lam, acc, eps_inf, eps_2, sig:weak ratio")
for l in lam:
    print(l, f(ms([s["L1"][l]["acc"] for s in MAIN])), f(ms([s["L1"][l]["inf"] for s in MAIN])),
          f(ms([s["L1"][l]["l2"] for s in MAIN])), f(ms([s["L1"][l]["ratio"] for s in MAIN]), 1))

# ------------------------------------------------------------ Figure 5 (E5)
F = pickle.load(open("root_cause_factorial.pkl", "rb"))
dmus = sorted({k[0] for k in F})
dps = sorted({k[1] for k in F})
fig, ax = plt.subplots(figsize=(4.8, 3.1))
cols = {1.5: BLUE, 3.0: AQUA, 6.0: VIOLET}
for dp in dps:
    b = [np.mean([r["bayes"] for r in F[(d, dp)]]) for d in dmus]
    m = [ms([r["mlp"] for r in F[(d, dp)]]) for d in dmus]
    ax.loglog(dmus, b, "-", color=cols[dp], lw=1.6)
    ax.errorbar(dmus, [x[0] for x in m], yerr=[x[1] for x in m], fmt="o", color=cols[dp],
                mfc="white", ms=5, lw=1, capsize=2)
    ax.text(dmus[-1] * 1.06, b[-1] * {1.5: 1.0, 3.0: 1.12, 6.0: 0.86}[dp], rf"$d'={dp:g}$", color=cols[dp], fontsize=8, va="center")
ax.loglog(dmus, np.array(dmus) / 2, ":", color=MUTED, lw=1.2)
ax.text(dmus[1], dmus[1] / 2 * 0.55, r"dotted: $\Delta\mu/2$", color=MUTED, fontsize=8, ha="center")
ax.set_xlim(0.04, 1.25)
ax.set_xlabel(r"class-mean gap $\Delta\mu$ (scale)")
ax.set_ylabel(r"median $\ell_\infty$ flip radius")
fig.tight_layout()
fig.savefig("fig_rc_scale.pdf")
plt.close(fig)
print("== E5: dmu, then (bayes, mlp, eps/dmu, bayes acc) per d'")
for d in dmus:
    row = []
    for dp in dps:
        b = np.mean([r["bayes"] for r in F[(d, dp)]]); m = np.mean([r["mlp"] for r in F[(d, dp)]])
        row.append(f"{b:.3f}/{m:.3f}/{b/d:.3f}/acc{np.mean([r['bayes_acc'] for r in F[(d, dp)]]):.3f}")
    print(d, row)
print("max |bayes-mlp|:", max(abs(np.mean([r['bayes'] for r in v]) - np.mean([r['mlp'] for r in v])) for v in F.values()))
print("mlp acc min:", min(np.mean([r['acc'] for r in v]) for v in F.values()))
