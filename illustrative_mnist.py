"""A real adversarial example on MNIST for illustrative_paper.tex (fig_il_mnist_adv.pdf)."""
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import root_cause_realdata as rd

tr, te = rd.load()
torch.manual_seed(0)
idx = torch.randperm(len(tr[0]))[:10000]
xtr, ytr = tr[0][idx].to(rd.DEV), tr[1][idx].to(rd.DEV)
m = rd.train(xtr, ytr, 10)
xs, ys = te[0][:500].to(rd.DEV), te[1][:500].to(rd.DEV)


def pgd_flip(x, y, eps, steps=40):
    d = torch.zeros_like(x)
    for _ in range(steps):
        d.requires_grad_(True)
        nn_loss = torch.nn.functional.cross_entropy(m(x + d), y)
        g = torch.autograd.grad(nn_loss, d)[0]
        d = (d.detach() + eps / 4 * g.sign()).clamp(-eps, eps)
        with torch.no_grad():
            if m(x + d).argmax(1).item() != y.item():
                return d, True
    return d, False


rows = []
for i in range(len(xs)):
    x, y = xs[i:i + 1], ys[i:i + 1]
    with torch.no_grad():
        if m(x).argmax(1).item() != y.item():
            continue
    for eps in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
        d, ok = pgd_flip(x, y, eps)
        if ok:
            rows.append((i, eps, d)); break
    if len(rows) >= 40:
        break
rows = [r for r in rows if 0.04 <= r[1] <= 0.06][:2]
print([(r[0], r[1]) for r in rows])
fig, axs = plt.subplots(len(rows), 3, figsize=(7.2, 2.5 * len(rows)))
for r_, (i, eps, d) in enumerate(rows):
    x = xs[i:i + 1]
    with torch.no_grad():
        p0 = torch.softmax(m(x), 1)[0]; p1 = torch.softmax(m(x + d), 1)[0]
    for c, (im, t) in enumerate([(x, f"original: '{p0.argmax().item()}' ({p0.max().item():.0%} sure)"),
                                 (d, f"the change (each pixel at most {eps:.2f})"),
                                 (x + d, f"changed: '{p1.argmax().item()}' ({p1.max().item():.0%} sure)")]):
        a = axs[r_, c]
        if c == 1:
            a.imshow(im.cpu().view(28, 28), cmap="RdBu_r", vmin=-0.1, vmax=0.1)
        else:
            a.imshow(im.cpu().view(28, 28) + 0.5, cmap="gray", vmin=0, vmax=1)
        a.set_title(t, fontsize=8); a.axis("off")
fig.tight_layout(); fig.savefig("fig_il_mnist_adv.pdf")
