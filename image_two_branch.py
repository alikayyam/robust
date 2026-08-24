"""
Appendix: the main text's image-domain disanalogy (Section 7.3) arises
because a single shared CNN sees the whole image, so "restrict the attack
to a region" is not the same as "restrict the attack to a channel" the way
it is in the vector task, where Model B has no input slot at all for the
noise dimensions. Here we build an architecture where that is no longer
true: a shape branch that only ever receives the cropped shape-region
pixels, and a background branch that only ever receives the image with
that same region zeroed out, concatenated before a shared classifier head.
Because each branch's weights only ever see its own region, restricting a
perturbation to one region now genuinely restricts it to one branch's
entire input, not just to a spatial subset of a shared model's input --
the real image-domain analog of Model A's per-dimension masking.
"""

import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F

from attacks import fgsm_success, pgd_success
from image_stimulus_experiment import (
    make_dataset, SHAPE_MASK, BG_MASK, SHAPE_BOX, IMG_SIZE, N_TRAIN, N_TEST, EPSILONS, SEEDS, N_SEEDS,
)

CROP = SHAPE_BOX[0].stop - SHAPE_BOX[0].start  # 20


class SmallConvHead(nn.Module):
    def __init__(self, in_size, feat_dim=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        pooled = in_size // 4
        self.fc = nn.Linear(16 * pooled * pooled, feat_dim)

    def forward(self, x):
        h = self.pool(F.relu(self.conv1(x)))
        h = self.pool(F.relu(self.conv2(h)))
        h = h.flatten(1)
        return F.relu(self.fc(h))


class TwoBranchCNN(nn.Module):
    """Shape branch sees only the cropped shape region (CROP x CROP);
    background branch sees the full image with that region zeroed. Both
    are derived from the same full-image tensor at forward time, but each
    branch's *weights* only ever have gradient/activation contact with its
    own region -- an architectural, not merely a masking, separation."""

    def __init__(self):
        super().__init__()
        self.shape_branch = SmallConvHead(CROP, feat_dim=16)
        self.bg_branch = SmallConvHead(IMG_SIZE, feat_dim=16)
        self.head = nn.Linear(32, 2)

    def forward(self, img):
        crop = img[:, :, SHAPE_BOX[0], SHAPE_BOX[1]]
        bg = img * BG_MASK
        fs = self.shape_branch(crop)
        fg = self.bg_branch(bg)
        return self.head(torch.cat([fs, fg], dim=1))


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


def ms(vals):
    t = torch.tensor(vals)
    return t.mean().item(), (t.std().item() if len(vals) > 1 else 0.0)


IMG_TWO_BRANCH_SEEDS = SEEDS[:5]  # 5 seeds: two forward passes per image (two branches) roughly
                                   # doubles training cost vs. the single-CNN Model A, so we use
                                   # half as many seeds as the main text's image experiment.

if __name__ == "__main__":
    accs = []
    fgsm_rows = {eps: {"shape": [], "bg": []} for eps in EPSILONS}
    pgd_rows = {eps: {"shape": [], "bg": []} for eps in EPSILONS}

    for s in IMG_TWO_BRANCH_SEEDS:
        torch.manual_seed(s)
        x_train, y_train = make_dataset(N_TRAIN)
        x_test, y_test = make_dataset(N_TEST)
        model = TwoBranchCNN()
        train(model, x_train, y_train)
        acc = accuracy(model, x_test, y_test)
        accs.append(acc)
        print(f"seed {s}: clean accuracy {acc:.4f}", flush=True)

        for eps in EPSILONS:
            fgsm_rows[eps]["shape"].append(fgsm_success(model, x_test, y_test, eps, mask=SHAPE_MASK)[0])
            fgsm_rows[eps]["bg"].append(fgsm_success(model, x_test, y_test, eps, mask=BG_MASK)[0])
            pgd_rows[eps]["shape"].append(pgd_success(model, x_test, y_test, eps, mask=SHAPE_MASK, steps=20, restarts=4)[0])
            pgd_rows[eps]["bg"].append(pgd_success(model, x_test, y_test, eps, mask=BG_MASK, steps=20, restarts=4)[0])

    acc_m, acc_s = ms(accs)
    print()
    print(f"Two-branch CNN clean accuracy: {acc_m:.4f}±{acc_s:.4f} (n={len(IMG_TWO_BRANCH_SEEDS)} seeds)")
    print(f"{'eps':>6s}  {'FGSM shape-branch':>18s}  {'FGSM bg-branch':>15s}  {'PGD shape-branch':>18s}  {'PGD bg-branch':>15s}")
    for eps in EPSILONS:
        fs_m, fs_s = ms(fgsm_rows[eps]["shape"]); fb_m, fb_s = ms(fgsm_rows[eps]["bg"])
        ps_m, ps_s = ms(pgd_rows[eps]["shape"]); pb_m, pb_s = ms(pgd_rows[eps]["bg"])
        print(f"{eps:6.2f}  {fs_m:.3f}±{fs_s:.3f}         {fb_m:.3f}±{fb_s:.3f}       "
              f"{ps_m:.3f}±{ps_s:.3f}         {pb_m:.3f}±{pb_s:.3f}")
