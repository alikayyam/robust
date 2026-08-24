"""
Image-domain analog of toy_robust_features_experiment.py: circle (class 0)
vs. square (class 1) drawn on a gray background whose shade is only
partially predictive of the class. Shape identity is the robust feature
(determines the label); background gray level is the non-robust feature
(correlated, not deterministic). A simple CNN is trained on the raw pixels
and probed with the same spatially-restricted attack idea: perturb only a
central box guaranteed to contain the shape ("shape region"), or only the
outer border guaranteed to be pure background ("background region").

All reported quantities are averaged over N_SEEDS independent random seeds
(data sampling, shape jitter, and model initialization); we report mean and
sample standard deviation across seeds. Default torch threading spawns a
thread pool per op whose dispatch overhead dwarfs the runtime of this small
CNN, so we pin to a single thread.
"""

import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

IMG_SIZE = 28
SHAPE_HALF = 8       # shape half-width/radius in pixels
JITTER = 2           # max random position jitter, pixels
FG_VALUE = 1.0        # shape (foreground) pixel intensity
BG_BASE = 0.35         # background gray level baseline
BG_ALPHA = 0.15        # background gray level shift per class
BG_SIGMA = 0.12        # background gray level noise std
PIXEL_NOISE = 0.35     # global pixel noise std (makes shape recognition imperfect)
N_TRAIN, N_TEST = 3000, 800
EPSILONS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))

# Fixed spatial masks: the shape (radius SHAPE_HALF, center jitter +-JITTER)
# always stays within [center-SHAPE_HALF-JITTER, center+SHAPE_HALF+JITTER].
# Anything strictly outside that box is guaranteed pure background in every
# image; we use that guaranteed region as the "background region" mask, and
# the inner box (which contains the shape plus some of its surrounding
# background) as the "shape region" mask.
_c = IMG_SIZE // 2
_margin = SHAPE_HALF + JITTER
SHAPE_BOX = (slice(_c - _margin, _c + _margin), slice(_c - _margin, _c + _margin))

def _region_masks():
    shape_mask = torch.zeros(IMG_SIZE, IMG_SIZE)
    shape_mask[SHAPE_BOX] = 1.0
    bg_mask = 1.0 - shape_mask
    return shape_mask, bg_mask

SHAPE_MASK, BG_MASK = _region_masks()


def make_dataset(n, pixel_noise=PIXEL_NOISE, draw_shape=True):
    y = torch.randint(0, 2, (n,))
    imgs = torch.full((n, 1, IMG_SIZE, IMG_SIZE), 0.0)

    bg = (BG_BASE + BG_ALPHA * (y.float() - 0.5) + BG_SIGMA * torch.randn(n)).clamp(0, 0.75)
    for i in range(n):
        imgs[i, 0, :, :] = bg[i]
        if draw_shape:
            cx = _c + torch.randint(-JITTER, JITTER + 1, (1,)).item()
            cy = _c + torch.randint(-JITTER, JITTER + 1, (1,)).item()
            yy, xx = torch.meshgrid(torch.arange(IMG_SIZE), torch.arange(IMG_SIZE), indexing="ij")
            if y[i].item() == 0:  # circle
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= SHAPE_HALF ** 2
            else:  # square
                mask = (xx - cx).abs().le(SHAPE_HALF) & (yy - cy).abs().le(SHAPE_HALF)
            imgs[i, 0][mask] = FG_VALUE

    if pixel_noise > 0:
        imgs = imgs + pixel_noise * torch.randn_like(imgs)
    imgs = imgs.clamp(0, 1)
    return imgs, y


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 2)

    def forward(self, x):
        h = self.pool(F.relu(self.conv1(x)))
        h = self.pool(F.relu(self.conv2(h)))
        h = h.flatten(1)
        return self.fc(h)


def train(model, x, y, epochs=15, lr=1e-3, batch_size=64, weight_decay=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
    return model


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()


def fgsm_success(model, x, y, epsilon, mask=None):
    """mask: (H, W) tensor of 1s (perturbable) / 0s, broadcast over channel/batch."""
    model.eval()
    with torch.no_grad():
        correct = model(x).argmax(dim=1) == y
    x_c, y_c = x[correct], y[correct]
    if len(x_c) == 0:
        return float("nan"), 0
    if mask is None:
        mask = torch.ones(IMG_SIZE, IMG_SIZE)
    x_adv = x_c.clone().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x_adv), y_c)
    loss.backward()
    x_adv = (x_c + epsilon * x_adv.grad.sign() * mask).clamp(0, 1).detach()
    with torch.no_grad():
        flipped = model(x_adv).argmax(dim=1) != y_c
    return flipped.float().mean().item(), len(x_c)


def run_seed(seed):
    """Trains the calibration models and Model A for one random seed and
    returns every metric reported by this experiment."""
    torch.manual_seed(seed)
    x_train, y_train = make_dataset(N_TRAIN)
    x_test, y_test = make_dataset(N_TEST)

    # shape-alone calibration: force background to a constant neutral value
    x_train_shapeonly, y_shapeonly = make_dataset(N_TRAIN, pixel_noise=0.0, draw_shape=True)
    x_test_shapeonly, y_test_shapeonly = make_dataset(N_TEST, pixel_noise=0.0, draw_shape=True)
    bg_const = 0.5
    x_train_shapeonly = torch.where(x_train_shapeonly < FG_VALUE - 1e-6, torch.tensor(bg_const), x_train_shapeonly)
    x_test_shapeonly = torch.where(x_test_shapeonly < FG_VALUE - 1e-6, torch.tensor(bg_const), x_test_shapeonly)
    x_train_shapeonly = (x_train_shapeonly + PIXEL_NOISE * torch.randn_like(x_train_shapeonly)).clamp(0, 1)
    x_test_shapeonly = (x_test_shapeonly + PIXEL_NOISE * torch.randn_like(x_test_shapeonly)).clamp(0, 1)

    model_shapeonly = SimpleCNN()
    train(model_shapeonly, x_train_shapeonly, y_shapeonly)
    acc_shapeonly = accuracy(model_shapeonly, x_test_shapeonly, y_test_shapeonly)

    # background-alone calibration: no shape drawn at all
    x_train_bgonly, y_bgonly = make_dataset(N_TRAIN, pixel_noise=PIXEL_NOISE, draw_shape=False)
    x_test_bgonly, y_test_bgonly = make_dataset(N_TEST, pixel_noise=PIXEL_NOISE, draw_shape=False)
    model_bgonly = SimpleCNN()
    train(model_bgonly, x_train_bgonly, y_bgonly)
    acc_bgonly = accuracy(model_bgonly, x_test_bgonly, y_test_bgonly)

    # Model A: both cues present
    model_a = SimpleCNN()
    train(model_a, x_train, y_train)
    acc_a = accuracy(model_a, x_test, y_test)

    rows = []
    for eps in EPSILONS:
        r_all, _ = fgsm_success(model_a, x_test, y_test, eps)
        r_shape, _ = fgsm_success(model_a, x_test, y_test, eps, mask=SHAPE_MASK)
        r_bg, _ = fgsm_success(model_a, x_test, y_test, eps, mask=BG_MASK)
        rows.append(dict(eps=eps, r_all=r_all, r_shape=r_shape, r_bg=r_bg))

    return dict(acc_shapeonly=acc_shapeonly, acc_bgonly=acc_bgonly, acc_a=acc_a, rows=rows)


def _mean_std(vals):
    t = torch.tensor(vals)
    std = t.std(unbiased=True).item() if len(vals) > 1 else 0.0
    return t.mean().item(), std


def aggregate_runs(runs):
    agg = {}
    for key in ("acc_shapeonly", "acc_bgonly", "acc_a"):
        agg[key] = _mean_std([r[key] for r in runs])
    agg_rows = []
    for i, eps in enumerate(EPSILONS):
        row = {"eps": eps}
        for col in ("r_all", "r_shape", "r_bg"):
            row[col] = _mean_std([r["rows"][i][col] for r in runs])
        agg_rows.append(row)
    agg["rows"] = agg_rows
    return agg


def _fmt(pair, decimals=3):
    m, s = pair
    return f"{m:.{decimals}f}±{s:.{decimals}f}"


if __name__ == "__main__":
    runs = [run_seed(s) for s in SEEDS]
    agg = aggregate_runs(runs)

    print(f"Averaged over {N_SEEDS} seeds (mean±std)")
    print("=== Calibration ===")
    print(f"  shape-alone accuracy (constant background): {_fmt(agg['acc_shapeonly'])}")
    print(f"  background-alone accuracy (no shape drawn): {_fmt(agg['acc_bgonly'])}")
    print()

    print("=== Model A (shape + background, both present) ===")
    print(f"  clean accuracy: {_fmt(agg['acc_a'])}")
    print()

    print(f"{'epsilon':>8s}  {'all pixels':>15s}  {'shape region':>15s}  {'background region':>18s}")
    for row in agg["rows"]:
        print(f"{row['eps']:8.2f}  {_fmt(row['r_all']):>15s}  {_fmt(row['r_shape']):>15s}  {_fmt(row['r_bg']):>18s}")
