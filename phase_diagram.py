"""
Appendix: the main text demonstrates the robust/non-robust effect at one
generative parameter setting (sigma_0=0.3, alpha=0.4, weight_decay=1e-2).
This sweeps signal reliability (sigma_0), spurious-feature strength
(alpha), and regularization strength (weight_decay) on a grid, training a
fresh Model-A-analog at each combination (5 seeds each), to characterize
where the effect appears vs. vanishes, rather than asserting it from one
point in the parameter space.
"""

import torch
torch.set_num_threads(1)
import torch.nn as nn

from attacks import fgsm_success
from toy_robust_features_experiment import MLP, train, per_input_weight_norm, accuracy, N_TRAIN, N_TEST

N_NOISE_DIMS = 5
SIGMA_NOISE = 0.25  # held fixed, as in the main text; only sigma0/alpha/weight_decay vary here
ATTACK_EPS = 0.2
SEEDS = list(range(5))

SIGMA0_GRID = [0.1, 0.2, 0.3, 0.4, 0.5]
ALPHA_GRID = [0.1, 0.2, 0.4, 0.6]
WD_GRID = [0.0, 1e-3, 1e-2, 1e-1]

SIGNAL_MASK = torch.tensor([1., 0, 0, 0, 0, 0])
NOISE_MASK = torch.tensor([0., 1, 1, 1, 1, 1])


def make_dataset_param(n, sigma0, alpha, sigma=SIGMA_NOISE):
    y = torch.randint(0, 2, (n,))
    x_signal = (y.float().unsqueeze(1) + sigma0 * torch.randn(n, 1)).clamp(0, 1)
    shift = (y.float().unsqueeze(1) - 0.5) * alpha
    x_noise = (0.5 + shift + sigma * torch.randn(n, N_NOISE_DIMS)).clamp(0, 1)
    return torch.cat([x_signal, x_noise], dim=1), y


def run_one(seed, sigma0, alpha, wd):
    torch.manual_seed(seed)
    x_train, y_train = make_dataset_param(N_TRAIN, sigma0, alpha)
    x_test, y_test = make_dataset_param(N_TEST, sigma0, alpha)
    model = MLP(in_dim=6)
    train(model, x_train, y_train, weight_decay=wd)

    acc = accuracy(model, x_test, y_test)
    norms = per_input_weight_norm(model).tolist()
    ratio = norms[0] / (sum(norms[1:]) / 5 + 1e-8)
    r_sig, _ = fgsm_success(model, x_test, y_test, ATTACK_EPS, mask=SIGNAL_MASK)
    r_noise, _ = fgsm_success(model, x_test, y_test, ATTACK_EPS, mask=NOISE_MASK)
    return dict(acc=acc, ratio=ratio, r_sig=r_sig, r_noise=r_noise)


def ms(vals):
    t = torch.tensor(vals)
    return t.mean().item(), (t.std().item() if len(vals) > 1 else 0.0)


if __name__ == "__main__":
    results = {}
    total = len(SIGMA0_GRID) * len(ALPHA_GRID) * len(WD_GRID)
    done = 0
    for sigma0 in SIGMA0_GRID:
        for alpha in ALPHA_GRID:
            for wd in WD_GRID:
                accs, ratios, r_sigs, r_noises = [], [], [], []
                for s in SEEDS:
                    r = run_one(s, sigma0, alpha, wd)
                    accs.append(r["acc"]); ratios.append(r["ratio"])
                    r_sigs.append(r["r_sig"]); r_noises.append(r["r_noise"])
                results[(sigma0, alpha, wd)] = dict(
                    acc=ms(accs), ratio=ms(ratios), r_sig=ms(r_sigs), r_noise=ms(r_noises),
                )
                done += 1
                print(f"[{done}/{total}] sigma0={sigma0} alpha={alpha} wd={wd}  "
                      f"acc={results[(sigma0,alpha,wd)]['acc'][0]:.3f}  "
                      f"ratio={results[(sigma0,alpha,wd)]['ratio'][0]:.2f}  "
                      f"r_sig={results[(sigma0,alpha,wd)]['r_sig'][0]:.3f}  "
                      f"r_noise={results[(sigma0,alpha,wd)]['r_noise'][0]:.3f}", flush=True)

    import pickle
    with open("phase_diagram_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved phase_diagram_results.pkl")
