"""
Regenerates Models A/B/C exactly as in toy_robust_features_experiment.py and
produces the paper's figures. Quantitative curves (epsilon curves, L1 sweep,
sparse curves, activation-sparsity sweep, actsparse curves) are averaged over
N_SEEDS seeds via toy_robust_features_experiment.run_seed/aggregate_runs, with
shaded bands showing +/-1 standard deviation across seeds. The two purely
illustrative figures (fig_example_attack.pdf: one concrete before/after
example; fig_architectures.pdf: a schematic diagram) use a single fixed seed,
since they show one representative instance rather than a statistic.
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_robust_features_experiment import (
    make_dataset, MLP, train, accuracy, fgsm_success, per_input_weight_norm,
    hidden_activation_stats, run_seed, aggregate_runs,
    N_TRAIN, N_TEST, EPSILONS, SEEDS, N_SEEDS, L1_LAMBDA, ACT_L1_LAMBDA,
)

runs = [run_seed(s) for s in SEEDS]
agg = aggregate_runs(runs)


def means(col):
    return [row[col][0] for row in agg["rows"]]


def stds(col):
    return [row[col][1] for row in agg["rows"]]


def band(ax, x, col, **kw):
    m, s = means(col), stds(col)
    lo = [mi - si for mi, si in zip(m, s)]
    hi = [mi + si for mi, si in zip(m, s)]
    ax.fill_between(x, lo, hi, alpha=0.15, **{k: v for k, v in kw.items() if k == "color"})
    return m


# ---------- Figure 1: attack success rate vs epsilon ----------
series_cols = {
    "A: all dims": "ra_all",
    "A: robust dim only": "ra_sig",
    "A: non-robust dims only": "ra_noise",
    "B: robust dim only": "rb",
    "C: all dims": "rc_all",
    "C: dead dims only": "rc_dead",
}
styles = {
    "A: all dims": dict(color="#c0392b", marker="o", linestyle="-"),
    "A: robust dim only": dict(color="#2980b9", marker="s", linestyle="-"),
    "A: non-robust dims only": dict(color="#e67e22", marker="^", linestyle="-"),
    "B: robust dim only": dict(color="#2980b9", marker="s", linestyle="--"),
    "C: all dims": dict(color="#27ae60", marker="D", linestyle="-"),
    "C: dead dims only": dict(color="#7f8c8d", marker="x", linestyle=":"),
}

fig, ax = plt.subplots(figsize=(6.2, 4.2))
series = {}
for name, col in series_cols.items():
    m = band(ax, EPSILONS, col, color=styles[name]["color"])
    series[name] = m
    ax.plot(EPSILONS, m, label=name, linewidth=1.8, markersize=5, **styles[name])
ax.set_xlabel(r"attack budget $\epsilon$ ($L_\infty$)")
ax.set_ylabel("attack success rate")
ax.set_ylim(-0.02, 0.75)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("fig_epsilon_curves.pdf")
plt.close(fig)

# ---------- Figure 2: one concrete example before/after attack ----------
# Illustrative only (single fixed seed) -- shows one representative attacked
# example, not a statistic, so seed variation across runs does not apply here.
def first_success_example(model, x, y, epsilon, mask=None):
    model.eval()
    with torch.no_grad():
        correct = model(x).argmax(dim=1) == y
    x_c, y_c = x[correct], y[correct]
    if mask is None:
        mask = torch.ones(x.shape[1])
    x_adv_in = x_c.clone().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(model(x_adv_in), y_c)
    loss.backward()
    x_adv = (x_c + epsilon * x_adv_in.grad.sign() * mask).clamp(0, 1).detach()
    with torch.no_grad():
        pred_before = model(x_c).argmax(dim=1)
        pred_after = model(x_adv).argmax(dim=1)
    flipped = (pred_after != y_c).nonzero().flatten()
    idx = flipped[0].item()
    return x_c[idx], x_adv[idx], y_c[idx].item(), pred_before[idx].item(), pred_after[idx].item()


torch.manual_seed(0)
xA_train, yA_train = make_dataset(N_TRAIN, include_noise=True)
xA_test, yA_test = make_dataset(N_TEST, include_noise=True)
model_a = MLP(in_dim=6)
train(model_a, xA_train, yA_train)

xB_train, yB_train = make_dataset(N_TRAIN, include_noise=False)
xB_test, yB_test = make_dataset(N_TEST, include_noise=False)
model_b = MLP(in_dim=1)
train(model_b, xB_train, yB_train)

noise_mask = torch.tensor([0., 1, 1, 1, 1, 1])

x_orig_a, x_adv_a, y_a, pb_a, pa_a = first_success_example(model_a, xA_test, yA_test, 0.3, mask=noise_mask)
x_orig_b, x_adv_b, y_b, pb_b, pa_b = first_success_example(model_b, xB_test, yB_test, 0.3)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

dims_a = ["signal", "noise1", "noise2", "noise3", "noise4", "noise5"]
xpos = range(len(dims_a))
w = 0.35
axes[0].bar([p - w/2 for p in xpos], x_orig_a.tolist(), width=w, label="original", color="#2980b9")
axes[0].bar([p + w/2 for p in xpos], x_adv_a.tolist(), width=w, label="adversarial", color="#e67e22")
axes[0].set_xticks(list(xpos))
axes[0].set_xticklabels(dims_a, rotation=30, ha="right")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("input value")
axes[0].set_title(f"Model A, noise-only attack ($\\epsilon$=0.3)\ntrue $y$={y_a}, pred before={pb_a}, pred after={pa_a}", fontsize=9)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3, axis="y")

dims_b = ["signal"]
axes[1].bar([-w/2], x_orig_b.tolist(), width=w, label="original", color="#2980b9")
axes[1].bar([w/2], x_adv_b.tolist(), width=w, label="adversarial", color="#e67e22")
axes[1].set_xticks([0])
axes[1].set_xticklabels(dims_b)
axes[1].set_xlim(-1, 1)
axes[1].set_ylim(0, 1)
axes[1].set_title(f"Model B, its only dim ($\\epsilon$=0.3)\ntrue $y$={y_b}, pred before={pb_b}, pred after={pa_b}", fontsize=9)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig("fig_example_attack.pdf")
plt.close(fig)

print("Wrote fig_epsilon_curves.pdf and fig_example_attack.pdf")
print(f"Sanity check against Table 2 (eps=0.3, mean±std over {N_SEEDS} seeds):",
      {k: f"{v[-1]:.3f}±{s[-1]:.3f}" for (k, v), s in zip(series.items(), [stds(c) for c in series_cols.values()])})


# ---------- Figure 3: network architecture diagrams ----------
def draw_mlp(ax, input_specs, n_hidden, n_out, title):
    """input_specs: list of (label, color, dashed:bool)."""
    n_in = len(input_specs)
    x_in, x_hid, x_out = 0.0, 1.0, 2.0
    in_y = [0.5 + i * (n_hidden - 1) / max(n_in - 1, 1) for i in range(n_in)] if n_in > 1 else [ (n_hidden - 1) / 2 ]
    hid_y = list(range(n_hidden))
    out_y = [ (n_hidden - 1) / 2 - 0.6, (n_hidden - 1) / 2 + 0.6 ]

    for (label, color, dashed), y in zip(input_specs, in_y):
        for hy in hid_y:
            style = dict(color="0.75", linewidth=0.4, zorder=1)
            if dashed:
                style["linestyle"] = (0, (2, 2))
                style["color"] = "0.85"
            ax.plot([x_in, x_hid], [y, hy], **style)
    for hy in hid_y:
        for oy in out_y:
            ax.plot([x_hid, x_out], [hy, oy], color="0.75", linewidth=0.4, zorder=1)

    for (label, color, dashed), y in zip(input_specs, in_y):
        edge = "0.4" if not dashed else "0.6"
        ls = "--" if dashed else "-"
        ax.scatter([x_in], [y], s=260, color=color, edgecolors=edge, linestyles=ls, zorder=2)
        ax.text(x_in, y, label, ha="center", va="center", fontsize=6, color="white", zorder=3)
    for hy in hid_y:
        ax.scatter([x_hid], [hy], s=220, color="#7fb3e0", edgecolors="0.4", zorder=2)
    for oy, lab in zip(out_y, ["0", "1"]):
        ax.scatter([x_out], [oy], s=260, color="0.6", edgecolors="0.3", zorder=2)
        ax.text(x_out, oy, lab, ha="center", va="center", fontsize=7, zorder=3)

    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(-1.2, n_hidden + 0.5)
    ax.axis("off")


fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))

specs_a = [("s", "#2e8b57", False)] + [(f"n{i}", "#e07b1a", False) for i in range(1, 6)]
draw_mlp(axes[0], specs_a, n_hidden=8, n_out=2, title="Model A: 6 real inputs")

specs_b = [("s", "#2e8b57", False)]
draw_mlp(axes[1], specs_b, n_hidden=8, n_out=2, title="Model B: signal dim. only")

specs_c = [("s", "#2e8b57", False)] + [("0", "#999999", True) for _ in range(5)]
draw_mlp(axes[2], specs_c, n_hidden=8, n_out=2, title="Model C: 5 dims zeroed")

fig.tight_layout()
fig.savefig("fig_architectures.pdf")
plt.close(fig)
print("Wrote fig_architectures.pdf")


# ---------- Figure 4: L1 sweep -- signal vs. noise weight norm ----------
l1_values = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]


def l1_sweep_run(seed):
    torch.manual_seed(seed)
    xA_tr, yA_tr = make_dataset(N_TRAIN, include_noise=True)
    xA_te, yA_te = make_dataset(N_TEST, include_noise=True)
    signal, noise, acc = [], [], []
    for l1 in l1_values:
        torch.manual_seed(seed)
        m = MLP(in_dim=6)
        train(m, xA_tr, yA_tr, weight_decay=0.0, l1_lambda=l1)
        norms = per_input_weight_norm(m).tolist()
        signal.append(norms[0])
        noise.append(sum(norms[1:]) / 5)
        acc.append(accuracy(m, xA_te, yA_te))
    return signal, noise, acc


l1_runs = [l1_sweep_run(s) for s in SEEDS]
sweep_signal_m, sweep_signal_s, sweep_noise_m, sweep_noise_s, sweep_acc_m, sweep_acc_s = [], [], [], [], [], []
for i in range(len(l1_values)):
    sig_vals = [r[0][i] for r in l1_runs]
    noise_vals = [r[1][i] for r in l1_runs]
    acc_vals = [r[2][i] for r in l1_runs]
    sig_t, noise_t, acc_t = torch.tensor(sig_vals), torch.tensor(noise_vals), torch.tensor(acc_vals)
    sweep_signal_m.append(sig_t.mean().item()); sweep_signal_s.append(sig_t.std().item())
    sweep_noise_m.append(noise_t.mean().item()); sweep_noise_s.append(noise_t.std().item())
    sweep_acc_m.append(acc_t.mean().item()); sweep_acc_s.append(acc_t.std().item())

fig, ax1 = plt.subplots(figsize=(6.0, 4.0))
ax1.errorbar(l1_values, sweep_signal_m, yerr=sweep_signal_s, marker="o", color="#2e8b57",
             label="signal weight norm", capsize=3)
ax1.errorbar(l1_values, sweep_noise_m, yerr=sweep_noise_s, marker="s", color="#e07b1a",
             label="mean noise weight norm", capsize=3)
ax1.set_xlabel(r"L1 penalty strength $\lambda_{L1}$")
ax1.set_ylabel("first-layer weight norm")
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.errorbar(l1_values, sweep_acc_m, yerr=sweep_acc_s, marker="^", color="0.4", linestyle="--",
             label="clean accuracy", capsize=3)
ax2.set_ylabel("clean test accuracy")
ax2.set_ylim(0.85, 1.0)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig("fig_l1_sweep.pdf")
plt.close(fig)
print("Wrote fig_l1_sweep.pdf")
print(f"L1 sweep data (mean±std over {N_SEEDS} seeds):", list(zip(
    l1_values,
    [f"{m:.3f}±{s:.3f}" for m, s in zip(sweep_signal_m, sweep_signal_s)],
    [f"{m:.3f}±{s:.3f}" for m, s in zip(sweep_noise_m, sweep_noise_s)],
    [f"{m:.3f}±{s:.3f}" for m, s in zip(sweep_acc_m, sweep_acc_s)],
)))

# ---------- Figure 5: Model A vs. A-sparse (L1=0.4) attack curves ----------
sparse_cols = {
    "A: all dims": "ra_all",
    "A-sparse: all dims": "ras_all",
    "A-sparse: robust dim only": "ras_sig",
    "A-sparse: non-robust dims only": "ras_noise",
    "B: robust dim only": "rb",
}
sparse_styles = {
    "A: all dims": dict(color="#c0392b", marker="o", linestyle="-"),
    "A-sparse: all dims": dict(color="#8e44ad", marker="D", linestyle="-"),
    "A-sparse: robust dim only": dict(color="#8e44ad", marker="s", linestyle=":"),
    "A-sparse: non-robust dims only": dict(color="#8e44ad", marker="^", linestyle="--"),
    "B: robust dim only": dict(color="#2980b9", marker="s", linestyle="--"),
}
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for name, col in sparse_cols.items():
    m = band(ax, EPSILONS, col, color=sparse_styles[name]["color"])
    ax.plot(EPSILONS, m, label=name, linewidth=1.8, markersize=5, **sparse_styles[name])
ax.set_xlabel(r"attack budget $\epsilon$ ($L_\infty$)")
ax.set_ylabel("attack success rate")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("fig_sparse_curves.pdf")
plt.close(fig)
print("Wrote fig_sparse_curves.pdf")


# ---------- Figure 6: activation-sparsity sweep ----------
act_l1_values = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 1.4, 1.6, 1.8, 2.0]


def act_sweep_run(seed):
    torch.manual_seed(seed)
    xA_tr, yA_tr = make_dataset(N_TRAIN, include_noise=True)
    xA_te, yA_te = make_dataset(N_TEST, include_noise=True)
    acc, meanact, ratio = [], [], []
    for lam in act_l1_values:
        torch.manual_seed(seed)
        m = MLP(in_dim=6)
        train(m, xA_tr, yA_tr, weight_decay=0.0, act_l1_lambda=lam)
        acc.append(accuracy(m, xA_te, yA_te))
        ma, _ = hidden_activation_stats(m, xA_te)
        meanact.append(ma)
        norms = per_input_weight_norm(m).tolist()
        ratio.append(norms[0] / (sum(norms[1:]) / 5))
    return acc, meanact, ratio


act_runs = [act_sweep_run(s) for s in SEEDS]
act_sweep_acc_m, act_sweep_acc_s = [], []
act_sweep_meanact_m, act_sweep_meanact_s = [], []
act_sweep_ratio_m, act_sweep_ratio_s = [], []
for i in range(len(act_l1_values)):
    acc_t = torch.tensor([r[0][i] for r in act_runs])
    ma_t = torch.tensor([r[1][i] for r in act_runs])
    ratio_t = torch.tensor([r[2][i] for r in act_runs])
    act_sweep_acc_m.append(acc_t.mean().item()); act_sweep_acc_s.append(acc_t.std().item())
    act_sweep_meanact_m.append(ma_t.mean().item()); act_sweep_meanact_s.append(ma_t.std().item())
    act_sweep_ratio_m.append(ratio_t.mean().item()); act_sweep_ratio_s.append(ratio_t.std().item())

fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
ax1.errorbar(act_l1_values, act_sweep_meanact_m, yerr=act_sweep_meanact_s, marker="o", color="#16a085",
             label="mean hidden activation", capsize=3)
ax1.errorbar(act_l1_values, act_sweep_ratio_m, yerr=act_sweep_ratio_s, marker="s", color="#d35400",
             label="signal:noise weight ratio", capsize=3)
ax1.set_xlabel(r"activation L1 penalty strength $\lambda_{act}$")
ax1.set_ylabel("mean activation / weight ratio")
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.errorbar(act_l1_values, act_sweep_acc_m, yerr=act_sweep_acc_s, marker="^", color="0.4", linestyle="--",
             label="clean accuracy", capsize=3)
ax2.set_ylabel("clean test accuracy")
ax2.axvline(1.7, color="0.7", linestyle=":", linewidth=1)
ax2.text(1.72, 0.6, "collapse", fontsize=8, color="0.5", rotation=90, va="bottom")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center left")
fig.tight_layout()
fig.savefig("fig_actsparse_sweep.pdf")
plt.close(fig)
print("Wrote fig_actsparse_sweep.pdf")
print(f"act sweep data (mean±std over {N_SEEDS} seeds):", list(zip(
    act_l1_values,
    [f"{m:.3f}±{s:.3f}" for m, s in zip(act_sweep_acc_m, act_sweep_acc_s)],
    [f"{m:.4f}±{s:.4f}" for m, s in zip(act_sweep_meanact_m, act_sweep_meanact_s)],
    [f"{m:.2f}±{s:.2f}" for m, s in zip(act_sweep_ratio_m, act_sweep_ratio_s)],
)))

# ---------- Figure 7: Model A vs. A-actsparse attack curves ----------
actsparse_cols = {
    "A: all dims": "ra_all",
    "A: non-robust dims only": "ra_noise",
    "Aa: all dims": "raa_all",
    "Aa: non-robust dims only": "raa_noise",
}
actsparse_styles = {
    "A: all dims": dict(color="#c0392b", marker="o", linestyle="-"),
    "A: non-robust dims only": dict(color="#e67e22", marker="^", linestyle="-"),
    "Aa: all dims": dict(color="#16a085", marker="D", linestyle="-"),
    "Aa: non-robust dims only": dict(color="#16a085", marker="^", linestyle="--"),
}
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for name, col in actsparse_cols.items():
    m = band(ax, EPSILONS, col, color=actsparse_styles[name]["color"])
    ax.plot(EPSILONS, m, label=name, linewidth=1.8, markersize=5, **actsparse_styles[name])
ax.set_xlabel(r"attack budget $\epsilon$ ($L_\infty$)")
ax.set_ylabel("attack success rate")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("fig_actsparse_curves.pdf")
plt.close(fig)
print("Wrote fig_actsparse_curves.pdf")
