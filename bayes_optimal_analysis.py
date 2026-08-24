"""
Exact Bayes-optimal analysis for the synthetic task of Section 3. Since the
generative model is fully known (clipped Gaussians with a common variance per
class pair), the Bayes-optimal decision rule for any subset of input
dimensions is a closed-form log-likelihood-ratio test, computed exactly here
(no approximation beyond Monte Carlo sampling of the *inputs*; the decision
rule itself and its plugged-in likelihoods are exact, including the
boundary point masses created by clipping to [0,1]).

Answers two questions the weight-norm comparisons elsewhere in the paper
cannot: (1) how much accuracy is actually available from the noise
dimensions when they are combined optimally, rather than read off one at a
time (Table 1's single-dimension numbers); and (2) whether the trained
MLP's reliance on the noise channel, measured by permutation ablation
(a model-agnostic reliance measure, unlike a weight norm), is larger or
smaller than a statistically optimal classifier's reliance on that same
channel.
"""

import math
import torch
torch.set_num_threads(1)

from toy_robust_features_experiment import (
    make_dataset, MLP, train, accuracy, N_TRAIN, N_TEST,
    SIGMA_SIGNAL, ALPHA, SIGMA, SEEDS, N_SEEDS,
)

MU0_SIGNAL, MU1_SIGNAL = 0.0, 1.0
MU0_NOISE, MU1_NOISE = 0.5 - 0.5 * ALPHA, 0.5 + 0.5 * ALPHA  # 0.3, 0.7 for ALPHA=0.4

SQRT2 = math.sqrt(2.0)


def normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / SQRT2))


def llr_dim(x, mu0, mu1, sigma):
    """Exact log P(x|y=1) - log P(x|y=0) for one dimension x = clip(N(mu_y,
    sigma^2), 0, 1), handling the point masses clipping creates at 0 and 1
    exactly (rather than treating x as if it were never clipped)."""
    interior = ((x - mu0) ** 2 - (x - mu1) ** 2) / (2 * sigma ** 2)

    m0_at0 = normal_cdf(torch.tensor((0.0 - mu0) / sigma))
    m1_at0 = normal_cdf(torch.tensor((0.0 - mu1) / sigma))
    m0_at1 = 1.0 - normal_cdf(torch.tensor((1.0 - mu0) / sigma))
    m1_at1 = 1.0 - normal_cdf(torch.tensor((1.0 - mu1) / sigma))
    llr_at0 = torch.log(m1_at0 + 1e-12) - torch.log(m0_at0 + 1e-12)
    llr_at1 = torch.log(m1_at1 + 1e-12) - torch.log(m0_at1 + 1e-12)

    out = interior.clone()
    out = torch.where(x <= 0.0, llr_at0.expand_as(out), out)
    out = torch.where(x >= 1.0, llr_at1.expand_as(out), out)
    return out


def bayes_llr(x, dims):
    """Total log-likelihood ratio using only the given dims (0 = signal,
    1..5 = noise), summed (valid because dims are conditionally independent
    given y by construction of the generative model)."""
    total = torch.zeros(x.shape[0])
    for d in dims:
        if d == 0:
            total = total + llr_dim(x[:, 0], MU0_SIGNAL, MU1_SIGNAL, SIGMA_SIGNAL)
        else:
            total = total + llr_dim(x[:, d], MU0_NOISE, MU1_NOISE, SIGMA)
    return total


def bayes_predict(x, dims):
    return (bayes_llr(x, dims) > 0).long()


def bayes_accuracy(x, y, dims):
    return (bayes_predict(x, dims) == y).float().mean().item()


def permute_col(x, d, generator):
    x = x.clone()
    perm = torch.randperm(x.shape[0], generator=generator)
    x[:, d] = x[perm, d]
    return x


def permute_cols(x, dims, generator):
    x = x.clone()
    for d in dims:
        perm = torch.randperm(x.shape[0], generator=generator)
        x[:, d] = x[perm, d]
    return x


ALL_DIMS = list(range(6))
NOISE_DIMS = list(range(1, 6))


if __name__ == "__main__":
    # --- Part 1: population-level Bayes risk (large Monte Carlo sample; the
    # decision rule is exact, only the accuracy *estimate* has finite-sample
    # noise, which is negligible at this N: binomial SE <= 0.5/sqrt(N) ~ 0.001). ---
    torch.manual_seed(12345)
    N_BIG = 200_000
    x_big, y_big = make_dataset(N_BIG, include_noise=True)

    acc_full = bayes_accuracy(x_big, y_big, ALL_DIMS)
    acc_signal_only = bayes_accuracy(x_big, y_big, [0])
    acc_noise_only = bayes_accuracy(x_big, y_big, NOISE_DIMS)
    acc_one_noise = bayes_accuracy(x_big, y_big, [1])

    print(f"Monte Carlo N={N_BIG} (SE <~ 0.001 on any of these)")
    print(f"Bayes-optimal accuracy, all 6 dims:              {acc_full:.4f}")
    print(f"Bayes-optimal accuracy, x0 only:                 {acc_signal_only:.4f}")
    print(f"Bayes-optimal accuracy, 5 noise dims combined:   {acc_noise_only:.4f}")
    print(f"Bayes-optimal accuracy, 1 noise dim only:        {acc_one_noise:.4f}")
    print()

    # log-likelihood-ratio weight each channel carries per unit of feature
    # value, ignoring clipping (valid in the interior, i.e. away from 0/1) --
    # this is the exact linear-discriminant weight for a clipped-Gaussian
    # pair with common variance, and is what a *linear* Bayes-optimal rule
    # would assign; the actual rule above additionally exploits the boundary
    # atoms, but this is the right quantity to compare against the MLP's
    # first-layer weight norms elsewhere in the paper.
    w_signal = (MU1_SIGNAL - MU0_SIGNAL) / SIGMA_SIGNAL ** 2
    w_noise_each = (MU1_NOISE - MU0_NOISE) / SIGMA ** 2
    w_noise_combined = 5 * w_noise_each
    print(f"Bayes-optimal linear weight, signal dim:              {w_signal:.3f}")
    print(f"Bayes-optimal linear weight, one noise dim:           {w_noise_each:.3f}")
    print(f"Bayes-optimal linear weight, all 5 noise dims summed: {w_noise_combined:.3f}")
    print(f"  -> Bayes-optimal combined-noise:signal weight ratio: {w_noise_combined/w_signal:.3f}")
    print()

    # --- Part 2: permutation-importance reliance, Bayes-optimal vs trained
    # MLP (Model A), on the same 10 seeds/test sets used everywhere else in
    # the paper, so the comparison is apples-to-apples. ---
    gen = torch.Generator().manual_seed(999)

    bayes_drop_signal, bayes_drop_noise = [], []
    mlp_drop_signal, mlp_drop_noise = [], []
    mlp_acc_list = []

    for s in SEEDS:
        torch.manual_seed(s)
        xA_train, yA_train = make_dataset(N_TRAIN, include_noise=True)
        xA_test, yA_test = make_dataset(N_TEST, include_noise=True)
        model_a = MLP(in_dim=6)
        train(model_a, xA_train, yA_train)
        mlp_acc_list.append(accuracy(model_a, xA_test, yA_test))

        base_bayes = bayes_accuracy(xA_test, yA_test, ALL_DIMS)
        x_signal_perm = permute_col(xA_test, 0, gen)
        x_noise_perm = permute_cols(xA_test, NOISE_DIMS, gen)
        bayes_drop_signal.append(base_bayes - bayes_accuracy(x_signal_perm, yA_test, ALL_DIMS))
        bayes_drop_noise.append(base_bayes - bayes_accuracy(x_noise_perm, yA_test, ALL_DIMS))

        base_mlp = accuracy(model_a, xA_test, yA_test)
        mlp_drop_signal.append(base_mlp - accuracy(model_a, x_signal_perm, yA_test))
        mlp_drop_noise.append(base_mlp - accuracy(model_a, x_noise_perm, yA_test))

    def ms(vals):
        t = torch.tensor(vals)
        return t.mean().item(), t.std().item()

    bs_m, bs_s = ms(bayes_drop_signal)
    bn_m, bn_s = ms(bayes_drop_noise)
    ms_m, ms_s = ms(mlp_drop_signal)
    mn_m, mn_s = ms(mlp_drop_noise)

    print(f"Permutation-importance accuracy drop (mean±std over {N_SEEDS} seeds):")
    print(f"  Bayes-optimal, signal permuted:  {bs_m:.4f}±{bs_s:.4f}")
    print(f"  Bayes-optimal, noise permuted:   {bn_m:.4f}±{bn_s:.4f}")
    print(f"  Bayes-optimal noise:signal drop ratio: {bn_m/bs_m:.2f}")
    print(f"  MLP (Model A), signal permuted:  {ms_m:.4f}±{ms_s:.4f}")
    print(f"  MLP (Model A), noise permuted:   {mn_m:.4f}±{mn_s:.4f}")
    print(f"  MLP noise:signal drop ratio:     {mn_m/ms_m:.2f}")
    print()
    print(f"  MLP clean accuracy on these seeds: {sum(mlp_acc_list)/len(mlp_acc_list):.4f}")
