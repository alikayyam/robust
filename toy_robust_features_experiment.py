"""
Simplified test of one intuition: if we remove (not just zero-value, but
architecturally remove) the non-robust, merely-correlated input dimensions,
does the resulting model become more robust to a standard adversarial attack?

Model A: 6 inputs (1 deterministic "signal" dim + 5 probabilistically-correlated
         "noise" dims), trained and attacked normally with full access.
Model B: 1 input (the signal dim only) -- the 5 noise dims are removed from the
         architecture entirely, not merely zeroed, so there is no attack
         surface for them at all.

Both are attacked with an unrestricted FGSM attack over whatever inputs they
actually have. Epsilon is kept well below 0.5 -- at 0.5 an attacker could move
a [0,1]-bounded feature across half its entire range, which stops being a
meaningful test of "small perturbation" robustness regardless of which
feature is attacked.

All reported quantities are averaged over N_SEEDS independent random seeds
(each reseeding data sampling and model initialization); we report mean and
sample standard deviation across seeds. Default torch threading spawns a
thread pool per op whose dispatch overhead dwarfs the runtime of these tiny
models, so we pin to a single thread.
"""

import torch
torch.set_num_threads(1)
import torch.nn as nn

torch.manual_seed(0)

N_TRAIN, N_TEST = 4000, 1000
N_NOISE_DIMS = 5
ALPHA = 0.4          # correlation strength of each noise dim with the label
SIGMA = 0.25         # noise std for the noise dims
SIGMA_SIGNAL = 0.3   # noise std for the signal dim (must be >0, see earlier note:
                      # a noiseless signal dim lets the optimizer ignore the noise
                      # dims entirely, since it can drive loss to ~0 alone)
WEIGHT_DECAY = 1e-2  # without this, Adam can just grow the signal weight instead
                      # of spreading reliance across correlated inputs
EPSILONS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
L1_LAMBDA = 0.4
ACT_L1_LAMBDA = 1.4
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))


def make_dataset(n, include_noise=True):
    y = torch.randint(0, 2, (n,))
    x_signal = (y.float().unsqueeze(1) + SIGMA_SIGNAL * torch.randn(n, 1)).clamp(0, 1)
    if not include_noise:
        return x_signal, y
    shift = (y.float().unsqueeze(1) - 0.5) * ALPHA
    x_noise = (0.5 + shift + SIGMA * torch.randn(n, N_NOISE_DIMS)).clamp(0, 1)
    return torch.cat([x_signal, x_noise], dim=1), y


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)

    def forward_with_hidden(self, x):
        """Exposes the post-ReLU hidden activation, needed for an activation
        sparsity penalty. Does not change what forward(x) returns elsewhere."""
        h = self.net[1](self.net[0](x))
        out = self.net[2](h)
        return out, h


def train(model, x, y, epochs=500, lr=0.02, weight_decay=WEIGHT_DECAY, l1_lambda=0.0,
          act_l1_lambda=0.0, group_lasso_lambda=0.0):
    """l1_lambda: elementwise L1 penalty on the first-layer weights (feature
    selection, see Section 6.5). act_l1_lambda: L1 penalty on the post-ReLU
    hidden activation instead -- a sparse-coding-style penalty on how
    many/how strongly hidden units fire per example, not on which input
    weights exist. group_lasso_lambda: L2,1 (group Lasso) penalty, one L2
    norm per input column rather than one L1 term per weight -- the
    natural feature-selection penalty when whole input dimensions, not
    individual weights, are the unit of interest (Appendix C)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        if act_l1_lambda > 0:
            logits, hidden = model.forward_with_hidden(x)
        else:
            logits = model(x)
        loss = loss_fn(logits, y)
        if l1_lambda > 0:
            loss = loss + l1_lambda * model.net[0].weight.abs().sum()
        if act_l1_lambda > 0:
            loss = loss + act_l1_lambda * hidden.mean()
        if group_lasso_lambda > 0:
            loss = loss + group_lasso_lambda * model.net[0].weight.norm(dim=0).sum()
        loss.backward()
        opt.step()
    return model


def per_input_weight_norm(model):
    return model.net[0].weight.data.norm(dim=0)


def hidden_activation_stats(model, x):
    with torch.no_grad():
        _, h = model.forward_with_hidden(x)
    mean_activation = h.mean().item()
    frac_zero = (h == 0).float().mean().item()
    return mean_activation, frac_zero


def accuracy(model, x, y):
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()


def fgsm_success(model, x, y, epsilon, mask=None):
    """FGSM attack, optionally restricted to a subset of input dims via `mask`
    (a length-D tensor of 1s/0s). mask=None means unrestricted (all dims)."""
    model.eval()
    with torch.no_grad():
        correct = model(x).argmax(dim=1) == y
    x_c, y_c = x[correct], y[correct]
    if len(x_c) == 0:
        return float("nan"), 0
    if mask is None:
        mask = torch.ones(x.shape[1])
    x_adv = x_c.clone().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x_adv), y_c)
    loss.backward()
    x_adv = (x_c + epsilon * x_adv.grad.sign() * mask).clamp(0, 1).detach()
    with torch.no_grad():
        flipped = model(x_adv).argmax(dim=1) != y_c
    return flipped.float().mean().item(), len(x_c)


def run_seed(seed):
    """Trains all five models for one random seed and returns every metric
    reported by this experiment, keyed the same way across seeds so results
    can be stacked and aggregated (see aggregate_runs)."""
    torch.manual_seed(seed)

    xA_train, yA_train = make_dataset(N_TRAIN, include_noise=True)
    xA_test, yA_test = make_dataset(N_TEST, include_noise=True)
    model_a = MLP(in_dim=6)
    train(model_a, xA_train, yA_train)

    model_a_sparse = MLP(in_dim=6)
    train(model_a_sparse, xA_train, yA_train, weight_decay=0.0, l1_lambda=L1_LAMBDA)

    model_a_act = MLP(in_dim=6)
    train(model_a_act, xA_train, yA_train, weight_decay=0.0, act_l1_lambda=ACT_L1_LAMBDA)

    xB_train, yB_train = make_dataset(N_TRAIN, include_noise=False)
    xB_test, yB_test = make_dataset(N_TEST, include_noise=False)
    model_b = MLP(in_dim=1)
    train(model_b, xB_train, yB_train)

    xC_train, xC_test = xA_train.clone(), xA_test.clone()
    xC_train[:, 1:] = 0.0
    xC_test[:, 1:] = 0.0
    model_c = MLP(in_dim=6)
    train(model_c, xC_train, yA_train)

    signal_mask = torch.tensor([1., 0, 0, 0, 0, 0])
    noise_mask = torch.tensor([0., 1, 1, 1, 1, 1])

    mean_act, frac_zero = hidden_activation_stats(model_a_act, xA_test)

    rows = []
    for eps in EPSILONS:
        ra_all, _ = fgsm_success(model_a, xA_test, yA_test, eps)
        ra_sig, _ = fgsm_success(model_a, xA_test, yA_test, eps, mask=signal_mask)
        ra_noise, _ = fgsm_success(model_a, xA_test, yA_test, eps, mask=noise_mask)
        ras_all, _ = fgsm_success(model_a_sparse, xA_test, yA_test, eps)
        ras_sig, _ = fgsm_success(model_a_sparse, xA_test, yA_test, eps, mask=signal_mask)
        ras_noise, _ = fgsm_success(model_a_sparse, xA_test, yA_test, eps, mask=noise_mask)
        raa_all, _ = fgsm_success(model_a_act, xA_test, yA_test, eps)
        raa_sig, _ = fgsm_success(model_a_act, xA_test, yA_test, eps, mask=signal_mask)
        raa_noise, _ = fgsm_success(model_a_act, xA_test, yA_test, eps, mask=noise_mask)
        rb, _ = fgsm_success(model_b, xB_test, yB_test, eps)
        rc_all, _ = fgsm_success(model_c, xC_test, yA_test, eps)
        rc_dead, _ = fgsm_success(model_c, xC_test, yA_test, eps, mask=noise_mask)
        rows.append(dict(
            eps=eps, ra_all=ra_all, ra_sig=ra_sig, ra_noise=ra_noise,
            ras_all=ras_all, ras_sig=ras_sig, ras_noise=ras_noise,
            raa_all=raa_all, raa_sig=raa_sig, raa_noise=raa_noise,
            rb=rb, rc_all=rc_all, rc_dead=rc_dead,
        ))

    return dict(
        acc_a=accuracy(model_a, xA_test, yA_test),
        acc_a_sparse=accuracy(model_a_sparse, xA_test, yA_test),
        acc_a_act=accuracy(model_a_act, xA_test, yA_test),
        acc_b=accuracy(model_b, xB_test, yB_test),
        acc_c=accuracy(model_c, xC_test, yA_test),
        norms_a=per_input_weight_norm(model_a).tolist(),
        norms_a_sparse=per_input_weight_norm(model_a_sparse).tolist(),
        norms_a_act=per_input_weight_norm(model_a_act).tolist(),
        a_act_mean_activation=mean_act,
        a_act_frac_zero=frac_zero,
        rows=rows,
    )


def _mean_std(vals):
    t = torch.tensor(vals)
    std = t.std(unbiased=True).item() if len(vals) > 1 else 0.0
    return t.mean().item(), std


ACT_COLLAPSE_THRESHOLD = 0.8  # acc_a_act below this = a dead (dying-ReLU) network, not a
                               # graceful accuracy loss -- see run_seed's per-seed spread.
_ACT_COLS = ("raa_all", "raa_sig", "raa_noise")


def aggregate_runs(runs):
    """Stacks a list of run_seed(...) dicts into per-metric (mean, std) pairs,
    computed across seeds. `rows` (the per-epsilon attack table) is aggregated
    row-by-row, matched by epsilon. Model A-actsparse's attack-rate columns
    (raa_*) exclude any seed whose model_a_act collapsed to chance level
    (Section 6.6): a collapsed network has an identically-zero gradient, so
    FGSM applies zero perturbation and "succeeds" 0% of the time by
    construction -- a degenerate artifact, not a robustness measurement, that
    would otherwise dilute the aggregate. `n_act_collapsed` records how many
    seeds were excluded this way."""
    agg = {}
    for key in ("acc_a", "acc_a_sparse", "acc_a_act", "acc_b", "acc_c",
                "a_act_mean_activation", "a_act_frac_zero"):
        agg[key] = _mean_std([r[key] for r in runs])
    for key in ("norms_a", "norms_a_sparse", "norms_a_act"):
        agg[key] = [_mean_std([r[key][d] for r in runs]) for d in range(6)]

    collapsed = [r["acc_a_act"] < ACT_COLLAPSE_THRESHOLD for r in runs]
    agg["n_act_collapsed"] = sum(collapsed)

    agg_rows = []
    for i, eps in enumerate(EPSILONS):
        row = {"eps": eps}
        for col in ("ra_all", "ra_sig", "ra_noise", "ras_all", "ras_sig", "ras_noise",
                    "raa_all", "raa_sig", "raa_noise", "rb", "rc_all", "rc_dead"):
            if col in _ACT_COLS:
                vals = [r["rows"][i][col] for r, c in zip(runs, collapsed) if not c]
            else:
                vals = [r["rows"][i][col] for r in runs]
            row[col] = _mean_std(vals)
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
    print(f"Model A (6 inputs, incl. 5 non-robust dims)   clean accuracy: {_fmt(agg['acc_a'])}")
    print(f"Model A-sparse (L1={L1_LAMBDA}, no L2)         clean accuracy: {_fmt(agg['acc_a_sparse'])}")
    print(f"Model A-actsparse (act-L1={ACT_L1_LAMBDA})     clean accuracy: {_fmt(agg['acc_a_act'])}")
    print(f"Model B (1 input, non-robust dims removed)     clean accuracy: {_fmt(agg['acc_b'])}")
    print(f"Model C (6 slots, non-robust dims zeroed only) clean accuracy: {_fmt(agg['acc_c'])}")
    print()

    print("Per-input first-layer weight norm (dim 0 = signal, 1-5 = noise):")
    print(f"  Model A (L2):        {[_fmt(v) for v in agg['norms_a']]}")
    print(f"  Model A-sparse (L1): {[_fmt(v) for v in agg['norms_a_sparse']]}")
    print(f"  Model A-actsparse:   {[_fmt(v) for v in agg['norms_a_act']]}")
    print(f"  Model A-actsparse hidden activation: mean={_fmt(agg['a_act_mean_activation'], 4)}, "
          f"fraction exactly zero={_fmt(agg['a_act_frac_zero'])}")
    print(f"  Model A-actsparse: {agg['n_act_collapsed']}/{N_SEEDS} seeds collapsed to chance level "
          f"(dying ReLU); excluded from the Aa: * columns below")
    print()

    header = (f"{'epsilon':>8s}  {'A: all':>13s}  {'A: robust':>13s}  {'A: non-robust':>13s}  "
               f"{'As: all':>13s}  {'As: robust':>13s}  {'As: non-robust':>13s}  "
               f"{'Aa: all':>13s}  {'Aa: robust':>13s}  {'Aa: non-robust':>13s}  "
               f"{'B: robust':>13s}  {'C: all':>13s}  {'C: dead dims':>13s}")
    print(header)
    for row in agg["rows"]:
        print(f"{row['eps']:8.2f}  {_fmt(row['ra_all']):>13s}  {_fmt(row['ra_sig']):>13s}  {_fmt(row['ra_noise']):>13s}  "
              f"{_fmt(row['ras_all']):>13s}  {_fmt(row['ras_sig']):>13s}  {_fmt(row['ras_noise']):>13s}  "
              f"{_fmt(row['raa_all']):>13s}  {_fmt(row['raa_sig']):>13s}  {_fmt(row['raa_noise']):>13s}  "
              f"{_fmt(row['rb']):>13s}  {_fmt(row['rc_all']):>13s}  {_fmt(row['rc_dead']):>13s}")
