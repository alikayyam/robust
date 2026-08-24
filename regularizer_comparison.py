"""
Appendix: Section 6.5's L1 feature-selection result compared against L2 at
higher strengths, group Lasso (L2,1 norm per input column -- the
theoretically natural penalty when whole input dimensions are the unit of
interest, not individual weights), elastic net (L1 + L2 together), and
post-hoc magnitude pruning (no retraining), to check the L1 effect is not
an accident of using L1 specifically or of the accuracy cost paid.

Each method is swept over a strength grid; for each we report the
operating point whose clean accuracy is closest to Model A-sparse's
(L1, lambda=0.4) operating accuracy (~0.966), so the comparison is at
matched accuracy across methods, the same standard already used for the
L1-vs-B comparison in Section 6.5.
"""

import copy
import torch
torch.set_num_threads(1)

from attacks import fgsm_success
from toy_robust_features_experiment import (
    make_dataset, MLP, train, accuracy, per_input_weight_norm,
    N_TRAIN, N_TEST, SEEDS,
)

SIGNAL_MASK = torch.tensor([1., 0, 0, 0, 0, 0])
NOISE_MASK = torch.tensor([0., 1, 1, 1, 1, 1])
TARGET_ACC = 0.966  # Model A-sparse (L1=0.4)'s operating accuracy, Section 6.5
EPS = 0.3


def ms(vals):
    t = torch.tensor(vals)
    return t.mean().item(), (t.std().item() if len(vals) > 1 else 0.0)


def eval_model(model, x_test, y_test):
    acc = accuracy(model, x_test, y_test)
    norms = per_input_weight_norm(model).tolist()
    ratio = norms[0] / (sum(norms[1:]) / 5 + 1e-8)
    r_sig, _ = fgsm_success(model, x_test, y_test, EPS, mask=SIGNAL_MASK)
    r_noise, _ = fgsm_success(model, x_test, y_test, EPS, mask=NOISE_MASK)
    return acc, ratio, r_sig, r_noise


def sweep(name, strengths, train_kwargs_fn, seeds=SEEDS):
    """train_kwargs_fn(strength) -> dict of train() kwargs."""
    per_strength = []
    for strength in strengths:
        accs, ratios, r_sigs, r_noises = [], [], [], []
        for s in seeds:
            torch.manual_seed(s)
            x_train, y_train = make_dataset(N_TRAIN, include_noise=True)
            x_test, y_test = make_dataset(N_TEST, include_noise=True)
            model = MLP(in_dim=6)
            train(model, x_train, y_train, **train_kwargs_fn(strength))
            acc, ratio, r_sig, r_noise = eval_model(model, x_test, y_test)
            accs.append(acc); ratios.append(ratio); r_sigs.append(r_sig); r_noises.append(r_noise)
        per_strength.append(dict(strength=strength, acc=ms(accs), ratio=ms(ratios),
                                  r_sig=ms(r_sigs), r_noise=ms(r_noises)))
        print(f"  [{name}] strength={strength:<8}  acc={per_strength[-1]['acc'][0]:.3f}  "
              f"ratio={per_strength[-1]['ratio'][0]:.2f}  r_noise={per_strength[-1]['r_noise'][0]:.3f}",
              flush=True)
    best = min(per_strength, key=lambda r: abs(r["acc"][0] - TARGET_ACC))
    return per_strength, best


if __name__ == "__main__":
    print("=== L2 (weight decay) at multiple strengths ===")
    l2_sweep, l2_best = sweep(
        "L2", [1e-2, 5e-2, 1e-1, 2e-1, 4e-1, 8e-1],
        lambda s: dict(weight_decay=s, l1_lambda=0.0),
    )

    print("=== Group Lasso (L2,1 norm per input column) ===")
    gl_sweep, gl_best = sweep(
        "GroupLasso", [0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
        lambda s: dict(weight_decay=0.0, group_lasso_lambda=s),
    )

    print("=== Elastic net (L1 + L2 together, L1 fixed at 0.1, L2 swept) ===")
    en_sweep, en_best = sweep(
        "ElasticNet", [0.0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        lambda s: dict(weight_decay=s, l1_lambda=0.1),
    )

    print("=== Post-hoc magnitude pruning (no retraining), pruning k smallest-norm noise columns ===")
    prune_results = []
    for k in range(6):
        accs, ratios, r_sigs, r_noises = [], [], [], []
        for s in SEEDS:
            torch.manual_seed(s)
            x_train, y_train = make_dataset(N_TRAIN, include_noise=True)
            x_test, y_test = make_dataset(N_TEST, include_noise=True)
            model = MLP(in_dim=6)
            train(model, x_train, y_train)  # plain L2 baseline (Model A's own recipe)

            model_p = copy.deepcopy(model)
            norms = per_input_weight_norm(model_p)
            noise_idx = list(range(1, 6))
            order = sorted(noise_idx, key=lambda i: norms[i].item())
            prune_idx = order[:k]
            with torch.no_grad():
                if prune_idx:
                    model_p.net[0].weight.data[:, prune_idx] = 0.0
            acc, ratio, r_sig, r_noise = eval_model(model_p, x_test, y_test)
            accs.append(acc); ratios.append(ratio); r_sigs.append(r_sig); r_noises.append(r_noise)
        prune_results.append(dict(strength=k, acc=ms(accs), ratio=ms(ratios), r_sig=ms(r_sigs), r_noise=ms(r_noises)))
        print(f"  [Pruning] k={k}  acc={prune_results[-1]['acc'][0]:.3f}  "
              f"ratio={prune_results[-1]['ratio'][0]:.2f}  r_noise={prune_results[-1]['r_noise'][0]:.3f}", flush=True)
    prune_best = min(prune_results, key=lambda r: abs(r["acc"][0] - TARGET_ACC))

    print()
    print("=== Matched-accuracy comparison (target acc ~= 0.966, from L1 lambda=0.4) ===")
    for name, best in [("L2 (weight decay)", l2_best), ("Group Lasso", gl_best),
                        ("Elastic net", en_best), ("Pruning", prune_best)]:
        print(f"  {name:20s} strength={best['strength']}  acc={best['acc'][0]:.3f}±{best['acc'][1]:.3f}  "
              f"ratio={best['ratio'][0]:.2f}±{best['ratio'][1]:.2f}  "
              f"r_sig={best['r_sig'][0]:.3f}±{best['r_sig'][1]:.3f}  "
              f"r_noise={best['r_noise'][0]:.3f}±{best['r_noise'][1]:.3f}")

    import pickle
    with open("regularizer_comparison_results.pkl", "wb") as f:
        pickle.dump(dict(l2=l2_sweep, gl=gl_sweep, en=en_sweep, prune=prune_results,
                          l2_best=l2_best, gl_best=gl_best, en_best=en_best, prune_best=prune_best), f)
    print("Saved regularizer_comparison_results.pkl")
