"""Builds fig_phase_diagram.pdf from phase_diagram_results.pkl."""

import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase_diagram import SIGMA0_GRID, ALPHA_GRID, WD_GRID

with open("phase_diagram_results.pkl", "rb") as f:
    results = pickle.load(f)

WD_FIXED = 1e-2  # matches the main text's operating point

ratio_grid = np.zeros((len(SIGMA0_GRID), len(ALPHA_GRID)))
rnoise_grid = np.zeros((len(SIGMA0_GRID), len(ALPHA_GRID)))
acc_grid = np.zeros((len(SIGMA0_GRID), len(ALPHA_GRID)))
for i, sigma0 in enumerate(SIGMA0_GRID):
    for j, alpha in enumerate(ALPHA_GRID):
        r = results[(sigma0, alpha, WD_FIXED)]
        ratio_grid[i, j] = r["ratio"][0]
        rnoise_grid[i, j] = r["r_noise"][0]
        acc_grid[i, j] = r["acc"][0]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

im0 = axes[0].imshow(ratio_grid, cmap="RdBu_r", vmin=0, vmax=8, aspect="auto")
axes[0].set_title("signal:noise weight-norm ratio")
im1 = axes[1].imshow(rnoise_grid, cmap="viridis", vmin=0, vmax=0.3, aspect="auto")
axes[1].set_title(r"non-robust-only attack success, $\epsilon=0.2$")
im2 = axes[2].imshow(acc_grid, cmap="Greens", vmin=0.85, vmax=1.0, aspect="auto")
axes[2].set_title("clean accuracy")

for ax, im in zip(axes, [im0, im1, im2]):
    ax.set_xticks(range(len(ALPHA_GRID)))
    ax.set_xticklabels(ALPHA_GRID)
    ax.set_yticks(range(len(SIGMA0_GRID)))
    ax.set_yticklabels(SIGMA0_GRID)
    ax.set_xlabel(r"$\alpha$ (spurious-feature strength)")
    ax.set_ylabel(r"$\sigma_0$ (signal noise, i.e.\ unreliability)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# mark the main text's operating point: sigma0=0.3, alpha=0.4
for ax in axes:
    ax.scatter([ALPHA_GRID.index(0.4)], [SIGMA0_GRID.index(0.3)], marker="*", s=200,
               color="black", edgecolors="white", linewidths=1, zorder=5)

fig.suptitle(r"Phase diagram at weight decay $=10^{-2}$ (the main text's regularization strength); "
             r"$\bigstar$ = main text's operating point", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig_phase_diagram.pdf")
print("Wrote fig_phase_diagram.pdf")

# a compact summary of the weight-decay sweep's effect, for text discussion
print()
print("Weight-decay sweep at alpha=0.1 (weak spurious correlation), across sigma0:")
for sigma0 in SIGMA0_GRID:
    row = [results[(sigma0, 0.1, wd)]["r_noise"][0] for wd in WD_GRID]
    print(f"  sigma0={sigma0}: r_noise by wd {WD_GRID} = {[round(v,3) for v in row]}")
print()
print("Weight-decay sweep at alpha=0.4, across sigma0:")
for sigma0 in SIGMA0_GRID:
    row = [results[(sigma0, 0.4, wd)]["r_noise"][0] for wd in WD_GRID]
    print(f"  sigma0={sigma0}: r_noise by wd {WD_GRID} = {[round(v,3) for v in row]}")
