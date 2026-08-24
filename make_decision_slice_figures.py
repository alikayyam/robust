"""
1D visualizations for the vector (non-image) task: class-conditional
distributions overlaid with the trained model's decision function, sliced
along one input dimension at a time. Makes visible what Table 1's
single-dimension accuracy numbers and the eps=0.5 boundary-reachability
discussion (Section 6.2) otherwise only describe in text. Shows a single
representative trained model (fixed seed) purely to illustrate the shape of
the decision function; not a quantitative claim, so seed variation does not
apply here.
"""

import torch
torch.set_num_threads(1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_robust_features_experiment import make_dataset, MLP, train, N_TRAIN, N_TEST

torch.manual_seed(0)
xA_train, yA_train = make_dataset(N_TRAIN, include_noise=True)
xA_test, yA_test = make_dataset(N_TEST, include_noise=True)
model_a = MLP(in_dim=6)
train(model_a, xA_train, yA_train)

xB_train, yB_train = make_dataset(N_TRAIN, include_noise=False)
xB_test, yB_test = make_dataset(N_TEST, include_noise=False)
model_b = MLP(in_dim=1)
train(model_b, xB_train, yB_train)


def prob_class1(model, x):
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)[:, 1]


# ---------- Figure: Model B, single dimension, full picture ----------
grid = torch.linspace(0, 1, 401).unsqueeze(1)
p = prob_class1(model_b, grid)

# find decision boundary crossing (p crosses 0.5)
crossings = grid[:-1][(p[:-1] < 0.5) & (p[1:] >= 0.5)]
boundary = crossings[0].item() if len(crossings) else float("nan")

fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
ax2 = ax1.twinx()

x0_vals = xB_test[yB_test == 0, 0].numpy()
x1_vals = xB_test[yB_test == 1, 0].numpy()
ax2.hist(x0_vals, bins=40, range=(0, 1), alpha=0.35, color="#2980b9", label="class 0 (y=0) data")
ax2.hist(x1_vals, bins=40, range=(0, 1), alpha=0.35, color="#c0392b", label="class 1 (y=1) data")
ax2.set_ylabel("count")

ax1.plot(grid.squeeze().numpy(), p.numpy(), color="black", linewidth=2, label="model P(class=1)")
ax1.axvline(boundary, color="0.3", linestyle="--", linewidth=1)
ax1.text(boundary + 0.01, 0.05, f"boundary\n$\\approx${boundary:.3f}", fontsize=8)
ax1.axhline(0.5, color="0.7", linestyle=":", linewidth=1)
ax1.set_xlabel("signal dimension value")
ax1.set_ylabel("P(class = 1)")
ax1.set_ylim(-0.02, 1.02)

# illustrate the eps=0.5 "spans half the domain" point from Section 6.2
ax1.annotate("", xy=(boundary, 0.9), xytext=(0.0, 0.9),
             arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5))
ax1.text(0.02, 0.93, r"$\epsilon=0.5$ reaches the boundary from $x=0$", fontsize=8, color="#27ae60")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("fig_decision_1d_modelb.pdf")
plt.close(fig)
print(f"Wrote fig_decision_1d_modelb.pdf (boundary={boundary:.4f})")


# ---------- Figure: Model A, sliced along each of the 6 dimensions ----------
dim_names = ["signal (x0)", "noise (x1)", "noise (x2)", "noise (x3)", "noise (x4)", "noise (x5)"]
means = xA_test.mean(dim=0)  # baseline value for the dims NOT being swept

fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey=True)
for d in range(6):
    ax = axes[d // 3][d % 3]
    ax2 = ax.twinx()

    base = means.unsqueeze(0).repeat(401, 1)
    base[:, d] = torch.linspace(0, 1, 401)
    p = prob_class1(model_a, base)

    v0 = xA_test[yA_test == 0, d].numpy()
    v1 = xA_test[yA_test == 1, d].numpy()
    ax2.hist(v0, bins=30, range=(0, 1), alpha=0.3, color="#2980b9")
    ax2.hist(v1, bins=30, range=(0, 1), alpha=0.3, color="#c0392b")
    ax2.set_yticks([])

    ax.plot(torch.linspace(0, 1, 401).numpy(), p.numpy(), color="black", linewidth=1.8)
    ax.axhline(0.5, color="0.7", linestyle=":", linewidth=0.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(dim_names[d], fontsize=9)
    if d % 3 == 0:
        ax.set_ylabel("P(class = 1)")
    if d // 3 == 1:
        ax.set_xlabel("dimension value")

fig.suptitle("Model A: decision function sliced along each dimension\n"
              "(other dimensions held at their test-set mean; histograms show class-conditional data)",
              fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig_decision_1d_modela_allDims.pdf")
plt.close(fig)
print("Wrote fig_decision_1d_modela_allDims.pdf")
