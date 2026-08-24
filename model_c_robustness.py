"""
Follow-up on Model C (Section 6.4): is "the dead dimensions are not
exploitable" a general property of freezing inputs at zero, or an artifact
of those dimensions' weights happening to sit at a small, random
initialization? We test this directly rather than assuming it, by
deliberately constructing large-norm and adversarially-chosen dead-feature
weights, and by checking a linear model where the mechanism is fully
analytically transparent.

Because Model C's dead dimensions are always exactly 0 at input, their
first-layer weight gradient (an outer product with the input) is exactly
zero throughout training regardless of the weight's value -- so whatever
we set those weight columns to at initialization is exactly what they are
after "training." This lets us set them to whatever we like and ask
whether that choice determines exploitability.
"""

import torch
torch.set_num_threads(1)
import torch.nn as nn

from toy_robust_features_experiment import (
    make_dataset, MLP, train, accuracy, fgsm_success,
    N_TRAIN, N_TEST, EPSILONS, SEEDS, N_SEEDS,
)

NOISE_MASK = torch.tensor([0., 1, 1, 1, 1, 1])


def make_model_c(seed, dead_weight_fn):
    """dead_weight_fn(default_dead_weight) -> replacement dead_weight, applied
    to columns 1..5 of the first-layer weight before training. Because those
    input columns are always exactly 0, this value is preserved exactly
    through training."""
    torch.manual_seed(seed)
    xA_train, yA_train = make_dataset(N_TRAIN, include_noise=True)
    xA_test, yA_test = make_dataset(N_TEST, include_noise=True)
    xC_train, xC_test = xA_train.clone(), xA_test.clone()
    xC_train[:, 1:] = 0.0
    xC_test[:, 1:] = 0.0

    torch.manual_seed(seed)
    model_c = MLP(in_dim=6)
    with torch.no_grad():
        default_dead = model_c.net[0].weight.data[:, 1:6].clone()
        model_c.net[0].weight.data[:, 1:6] = dead_weight_fn(default_dead)
    train(model_c, xC_train, yA_train)
    return model_c, xC_test, yA_test


def eval_dead_dims(model, x_test, y_test, epsilons=EPSILONS):
    return [fgsm_success(model, x_test, y_test, e, mask=NOISE_MASK)[0] for e in epsilons]


class LinearModel(nn.Module):
    """A 2-class linear model (no hidden layer), for an analytically
    transparent version of the same question."""
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, x):
        return self.fc(x)


def make_linear_c(seed, dead_weight_fn, epochs=500, lr=0.05, weight_decay=1e-2):
    torch.manual_seed(seed)
    xA_train, yA_train = make_dataset(N_TRAIN, include_noise=True)
    xA_test, yA_test = make_dataset(N_TEST, include_noise=True)
    xC_train, xC_test = xA_train.clone(), xA_test.clone()
    xC_train[:, 1:] = 0.0
    xC_test[:, 1:] = 0.0

    torch.manual_seed(seed)
    model = LinearModel(in_dim=6)
    with torch.no_grad():
        default_dead = model.fc.weight.data[:, 1:6].clone()
        model.fc.weight.data[:, 1:6] = dead_weight_fn(default_dead)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xC_train), yA_train)
        loss.backward()
        opt.step()
    return model, xC_test, yA_test


def ms(vals):
    t = torch.tensor(vals)
    return t.mean().item(), (t.std().item() if len(vals) > 1 else 0.0)


if __name__ == "__main__":
    print("=== MLP Model C: dead-dims-only attack success, eps=0.3, by dead-weight construction ===")
    variants = {
        "default init (baseline)": lambda w: w,
        "3x init scale": lambda w: 3.0 * w,
        "10x init scale": lambda w: 10.0 * w,
        "100x init scale": lambda w: 100.0 * w,
    }
    for name, fn in variants.items():
        accs, dead_at_03 = [], []
        for s in SEEDS:
            model_c, xC_test, yA_test = make_model_c(s, fn)
            accs.append(accuracy(model_c, xC_test, yA_test))
            dead_at_03.append(fgsm_success(model_c, xC_test, yA_test, 0.3, mask=NOISE_MASK)[0])
        acc_m, acc_s = ms(accs)
        d_m, d_s = ms(dead_at_03)
        print(f"  {name:28s} clean acc={acc_m:.3f}±{acc_s:.3f}  dead-dims attack@0.3={d_m:.4f}±{d_s:.4f}")

    print()
    print("=== MLP Model C: adversarial dead weight = matched-seed trained Model A's noise-dim weights ===")
    accs, dead_curves = [], []
    for s in SEEDS:
        torch.manual_seed(s)
        xA_train, yA_train = make_dataset(N_TRAIN, include_noise=True)
        _, _ = make_dataset(N_TEST, include_noise=True)
        model_a = MLP(in_dim=6)
        train(model_a, xA_train, yA_train)
        adversarial_dead = model_a.net[0].weight.data[:, 1:6].clone()

        model_c, xC_test, yA_test = make_model_c(s, lambda w, adv=adversarial_dead: adv)
        accs.append(accuracy(model_c, xC_test, yA_test))
        dead_curves.append(eval_dead_dims(model_c, xC_test, yA_test))
    acc_m, acc_s = ms(accs)
    print(f"  clean acc={acc_m:.3f}±{acc_s:.3f}")
    for i, eps in enumerate(EPSILONS):
        d_m, d_s = ms([c[i] for c in dead_curves])
        print(f"  eps={eps:.2f}  dead-dims attack success={d_m:.4f}±{d_s:.4f}")

    print()
    print("=== Linear model (no hidden layer) analog, same three constructions, eps=0.3 ===")
    lin_variants = {
        "default init": lambda w: w,
        "10x init scale": lambda w: 10.0 * w,
    }
    for name, fn in lin_variants.items():
        accs, dead_at_03 = [], []
        for s in SEEDS:
            model, xC_test, yA_test = make_linear_c(s, fn)
            accs.append(accuracy(model, xC_test, yA_test))
            dead_at_03.append(fgsm_success(model, xC_test, yA_test, 0.3, mask=NOISE_MASK)[0])
        acc_m, acc_s = ms(accs)
        d_m, d_s = ms(dead_at_03)
        print(f"  {name:28s} clean acc={acc_m:.3f}±{acc_s:.3f}  dead-dims attack@0.3={d_m:.4f}±{d_s:.4f}")

    accs, dead_at_03, weight_norms = [], [], []
    for s in SEEDS:
        torch.manual_seed(s)
        xA_train, yA_train = make_dataset(N_TRAIN, include_noise=True)
        model_a_lin = LinearModel(in_dim=6)
        opt = torch.optim.Adam(model_a_lin.parameters(), lr=0.05, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(500):
            opt.zero_grad()
            loss = loss_fn(model_a_lin(xA_train), yA_train)
            loss.backward()
            opt.step()
        adversarial_dead_lin = model_a_lin.fc.weight.data[:, 1:6].clone()
        weight_norms.append(adversarial_dead_lin.norm(dim=0).mean().item())

        model, xC_test, yA_test = make_linear_c(s, lambda w, adv=adversarial_dead_lin: adv)
        accs.append(accuracy(model, xC_test, yA_test))
        dead_at_03.append(fgsm_success(model, xC_test, yA_test, 0.3, mask=NOISE_MASK)[0])
    acc_m, acc_s = ms(accs)
    d_m, d_s = ms(dead_at_03)
    wn_m, wn_s = ms(weight_norms)
    print(f"  {'adversarial (linear-A weights)':28s} clean acc={acc_m:.3f}±{acc_s:.3f}  "
          f"dead-dims attack@0.3={d_m:.4f}±{d_s:.4f}  (mean adversarial weight norm={wn_m:.3f}±{wn_s:.3f})")
