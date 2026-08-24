"""
Shared attack implementations used across the appendix experiments. FGSM
(used throughout the main text) is a single gradient-sign step; PGD takes
many smaller steps, each projected back into the L_inf ball around the
original point, and is the standard, harder attack the main text's FGSM-only
evaluation does not rule out. Works for both the vector task (mask shape
(D,)) and the image task (mask shape (H,W)) since the mask just broadcasts
against the trailing dimensions of x.
"""

import torch
import torch.nn as nn


def fgsm_success(model, x, y, epsilon, mask=None):
    model.eval()
    with torch.no_grad():
        correct = model(x).argmax(dim=1) == y
    x_c, y_c = x[correct], y[correct]
    if len(x_c) == 0:
        return float("nan"), 0
    if mask is None:
        mask = torch.ones_like(x_c[0])
    x_adv = x_c.clone().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x_adv), y_c)
    loss.backward()
    x_adv = (x_c + epsilon * x_adv.grad.sign() * mask).clamp(0, 1).detach()
    with torch.no_grad():
        flipped = model(x_adv).argmax(dim=1) != y_c
    return flipped.float().mean().item(), len(x_c)


def pgd_success(model, x, y, epsilon, mask=None, steps=20, step_size=None, restarts=1):
    """L_inf PGD with `restarts` random starts (best-of over restarts per
    point). step_size defaults to epsilon/4, a standard choice."""
    model.eval()
    with torch.no_grad():
        correct = model(x).argmax(dim=1) == y
    x_c, y_c = x[correct], y[correct]
    n = len(x_c)
    if n == 0:
        return float("nan"), 0
    if mask is None:
        mask = torch.ones_like(x_c[0])
    if step_size is None:
        step_size = epsilon / 4.0

    ever_flipped = torch.zeros(n, dtype=torch.bool)
    for r in range(restarts):
        if r == 0:
            delta = torch.zeros_like(x_c)
        else:
            delta = (torch.rand_like(x_c) * 2 - 1) * epsilon * mask
        x_adv = (x_c + delta).clamp(0, 1).detach()
        for _ in range(steps):
            x_adv.requires_grad_(True)
            loss = nn.functional.cross_entropy(model(x_adv), y_c)
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv + step_size * grad.sign() * mask
                x_adv = torch.max(torch.min(x_adv, x_c + epsilon), x_c - epsilon)
                x_adv = x_adv.clamp(0, 1)
        with torch.no_grad():
            ever_flipped |= (model(x_adv).argmax(dim=1) != y_c)
    return ever_flipped.float().mean().item(), n
