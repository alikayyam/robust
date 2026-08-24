"""
Figures for the image-domain (circle vs. square) extension: example stimuli,
and a concrete before/after comparison of a shape-region attack vs. a
background-region attack on the same image. Both figures show a single
representative instance (fixed seed) rather than a statistic, so seed
variation does not apply here; the quantitative attack-rate table is
produced separately by image_stimulus_experiment.py, averaged over seeds.
"""

import torch
torch.set_num_threads(1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from image_stimulus_experiment import (
    make_dataset, SimpleCNN, train, accuracy, fgsm_success,
    SHAPE_MASK, BG_MASK, IMG_SIZE, BG_BASE, BG_ALPHA, BG_SIGMA,
    SHAPE_HALF, JITTER, FG_VALUE, PIXEL_NOISE,
)

torch.manual_seed(0)
N_TRAIN, N_TEST = 3000, 800
x_train, y_train = make_dataset(N_TRAIN)
x_test, y_test = make_dataset(N_TEST)
model_a = SimpleCNN()
train(model_a, x_train, y_train)
print("Model A clean accuracy:", accuracy(model_a, x_test, y_test))

# ---------- Figure: clean (noise-free) prototypes vs. the actual noisy stimuli ----------
# Purely illustrative (fixed seed, hand-constructed): shows what the underlying
# shape/background signal looks like with each noise source removed one at a
# time, since the fully-noisy training stimuli (bottom row) are hard to parse
# by eye even though the CNN classifies them at ~99.6% accuracy (Section 7.2's
# point that convolution spatially averages out pixel noise).
def draw_shape_image(label, bg_value, pixel_noise=0.0, jitter=False):
    img = torch.full((1, IMG_SIZE, IMG_SIZE), bg_value)
    cx = cy = IMG_SIZE // 2
    if jitter:
        cx = cx + torch.randint(-JITTER, JITTER + 1, (1,)).item()
        cy = cy + torch.randint(-JITTER, JITTER + 1, (1,)).item()
    yy, xx = torch.meshgrid(torch.arange(IMG_SIZE), torch.arange(IMG_SIZE), indexing="ij")
    if label == 0:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= SHAPE_HALF ** 2
    else:
        mask = (xx - cx).abs().le(SHAPE_HALF) & (yy - cy).abs().le(SHAPE_HALF)
    img[0][mask] = FG_VALUE
    if pixel_noise > 0:
        img = img + pixel_noise * torch.randn_like(img)
    return img.clamp(0, 1)

torch.manual_seed(2)
bg_mean = {0: BG_BASE + BG_ALPHA * (0 - 0.5), 1: BG_BASE + BG_ALPHA * (1 - 0.5)}
clean = {lab: draw_shape_image(lab, bg_mean[lab])[0] for lab in (0, 1)}
bg_noise_only = {lab: draw_shape_image(lab, bg_mean[lab] + BG_SIGMA * torch.randn(1).item())[0] for lab in (0, 1)}
full_noisy = {0: x_test[(y_test == 0).nonzero()[0].item(), 0], 1: x_test[(y_test == 1).nonzero()[0].item(), 0]}

fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.2))
col_titles = ["clean prototype\n(no noise at all)",
              "+ background noise only\n(no pixel noise, no jitter)",
              "actual training stimulus\n(full noise, as used)"]
for row, lab in enumerate((0, 1)):
    for col, img in enumerate((clean[lab], bg_noise_only[lab], full_noisy[lab])):
        ax = axes[row][col]
        ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if row == 0:
            ax.set_title(col_titles[col], fontsize=8)
    axes[row][0].text(-0.15, 0.5, "circle" if lab == 0 else "square", fontsize=9,
                       rotation=90, va="center", ha="center", transform=axes[row][0].transAxes)
fig.tight_layout()
fig.savefig("fig_clean_samples.pdf")
plt.close(fig)
print("Wrote fig_clean_samples.pdf")

# ---------- Figure: example stimuli ----------
fig, axes = plt.subplots(2, 4, figsize=(9, 4.6))
shown = {0: 0, 1: 0}
i = 0
while (shown[0] < 4 or shown[1] < 4) and i < len(x_test):
    label = y_test[i].item()
    if shown[label] < 4:
        ax = axes[label][shown[label]]
        ax.imshow(x_test[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"class {label} ({'circle' if label==0 else 'square'})", fontsize=8)
        ax.axis("off")
        shown[label] += 1
    i += 1
fig.tight_layout()
fig.savefig("fig_stimulus_examples.pdf")
plt.close(fig)
print("Wrote fig_stimulus_examples.pdf")

# ---------- Figure: shape-region vs background-region attack, same image ----------
def first_success_example(model, x, y, epsilon, mask):
    model.eval()
    with torch.no_grad():
        correct = model(x).argmax(dim=1) == y
    x_c, y_c = x[correct], y[correct]
    x_adv_in = x_c.clone().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(model(x_adv_in), y_c)
    loss.backward()
    x_adv = (x_c + epsilon * x_adv_in.grad.sign() * mask).clamp(0, 1).detach()
    with torch.no_grad():
        pred_before = model(x_c).argmax(dim=1)
        pred_after = model(x_adv).argmax(dim=1)
    flipped = (pred_after != y_c).nonzero().flatten()
    idx = flipped[0].item()
    return x_c[idx, 0], x_adv[idx, 0], y_c[idx].item(), pred_before[idx].item(), pred_after[idx].item()

EPS = 0.2
orig_s, adv_s, y_s, pb_s, pa_s = first_success_example(model_a, x_test, y_test, EPS, SHAPE_MASK)
orig_b, adv_b, y_b, pb_b, pa_b = first_success_example(model_a, x_test, y_test, EPS, BG_MASK)

fig, axes = plt.subplots(2, 2, figsize=(6.2, 6.4))
axes[0][0].imshow(orig_s.numpy(), cmap="gray", vmin=0, vmax=1)
axes[0][0].set_title(f"original (true y={y_s}, pred={pb_s})", fontsize=8)
axes[0][1].imshow(adv_s.numpy(), cmap="gray", vmin=0, vmax=1)
axes[0][1].set_title(f"shape-region attack ($\\epsilon$={EPS})\npred={pa_s}", fontsize=8)
axes[1][0].imshow(orig_b.numpy(), cmap="gray", vmin=0, vmax=1)
axes[1][0].set_title(f"original (true y={y_b}, pred={pb_b})", fontsize=8)
axes[1][1].imshow(adv_b.numpy(), cmap="gray", vmin=0, vmax=1)
axes[1][1].set_title(f"background-region attack ($\\epsilon$={EPS})\npred={pa_b}", fontsize=8)
for row in axes:
    for ax in row:
        ax.axis("off")
fig.tight_layout()
fig.savefig("fig_image_attack_example.pdf")
plt.close(fig)
print("Wrote fig_image_attack_example.pdf")
