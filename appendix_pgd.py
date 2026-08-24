"""
Appendix: does a stronger attack (PGD, multi-step) change the main text's
conclusions, which use single-step FGSM throughout? Re-attacks the same
trained models (vector task: Model A/B/C; image task: Model A) with 20-step
L_inf PGD (4 random restarts) at the same epsilons, averaged over the same
seeds, and reports FGSM alongside PGD for direct comparison.
"""

import torch
torch.set_num_threads(1)

from attacks import fgsm_success, pgd_success
from toy_robust_features_experiment import (
    make_dataset, MLP, train, N_TRAIN, N_TEST, EPSILONS, SEEDS, N_SEEDS,
)

SIGNAL_MASK = torch.tensor([1., 0, 0, 0, 0, 0])
NOISE_MASK = torch.tensor([0., 1, 1, 1, 1, 1])
PGD_STEPS, PGD_RESTARTS = 20, 4


def ms(vals):
    t = torch.tensor(vals)
    return t.mean().item(), (t.std().item() if len(vals) > 1 else 0.0)


def run_vector():
    rows = {eps: {"fgsm": {"a_all": [], "a_sig": [], "a_noise": [], "b": [], "c_all": [], "c_dead": []},
                  "pgd": {"a_all": [], "a_sig": [], "a_noise": [], "b": [], "c_all": [], "c_dead": []}}
            for eps in EPSILONS}

    for s in SEEDS:
        torch.manual_seed(s)
        xA_train, yA_train = make_dataset(N_TRAIN, include_noise=True)
        xA_test, yA_test = make_dataset(N_TEST, include_noise=True)
        model_a = MLP(in_dim=6)
        train(model_a, xA_train, yA_train)

        xB_train, yB_train = make_dataset(N_TRAIN, include_noise=False)
        xB_test, yB_test = make_dataset(N_TEST, include_noise=False)
        model_b = MLP(in_dim=1)
        train(model_b, xB_train, yB_train)

        xC_train, xC_test = xA_train.clone(), xA_test.clone()
        xC_train[:, 1:] = 0.0
        xC_test[:, 1:] = 0.0
        model_c = MLP(in_dim=6)
        train(model_c, xC_train, yA_train)

        for eps in EPSILONS:
            rows[eps]["fgsm"]["a_all"].append(fgsm_success(model_a, xA_test, yA_test, eps)[0])
            rows[eps]["fgsm"]["a_sig"].append(fgsm_success(model_a, xA_test, yA_test, eps, mask=SIGNAL_MASK)[0])
            rows[eps]["fgsm"]["a_noise"].append(fgsm_success(model_a, xA_test, yA_test, eps, mask=NOISE_MASK)[0])
            rows[eps]["fgsm"]["b"].append(fgsm_success(model_b, xB_test, yB_test, eps)[0])
            rows[eps]["fgsm"]["c_all"].append(fgsm_success(model_c, xC_test, yA_test, eps)[0])
            rows[eps]["fgsm"]["c_dead"].append(fgsm_success(model_c, xC_test, yA_test, eps, mask=NOISE_MASK)[0])

            rows[eps]["pgd"]["a_all"].append(pgd_success(model_a, xA_test, yA_test, eps, steps=PGD_STEPS, restarts=PGD_RESTARTS)[0])
            rows[eps]["pgd"]["a_sig"].append(pgd_success(model_a, xA_test, yA_test, eps, mask=SIGNAL_MASK, steps=PGD_STEPS, restarts=PGD_RESTARTS)[0])
            rows[eps]["pgd"]["a_noise"].append(pgd_success(model_a, xA_test, yA_test, eps, mask=NOISE_MASK, steps=PGD_STEPS, restarts=PGD_RESTARTS)[0])
            rows[eps]["pgd"]["b"].append(pgd_success(model_b, xB_test, yB_test, eps, steps=PGD_STEPS, restarts=PGD_RESTARTS)[0])
            rows[eps]["pgd"]["c_all"].append(pgd_success(model_c, xC_test, yA_test, eps, steps=PGD_STEPS, restarts=PGD_RESTARTS)[0])
            rows[eps]["pgd"]["c_dead"].append(pgd_success(model_c, xC_test, yA_test, eps, mask=NOISE_MASK, steps=PGD_STEPS, restarts=PGD_RESTARTS)[0])

    print(f"Vector task, FGSM vs PGD-{PGD_STEPS} ({PGD_RESTARTS} restarts), mean±std over {N_SEEDS} seeds")
    header = f"{'eps':>6s}  {'attack':>6s}  {'A:all':>13s}  {'A:sig':>13s}  {'A:noise':>13s}  {'B':>13s}  {'C:all':>13s}  {'C:dead':>13s}"
    print(header)
    for eps in EPSILONS:
        for kind in ("fgsm", "pgd"):
            vals = {k: ms(v) for k, v in rows[eps][kind].items()}
            print(f"{eps:6.2f}  {kind:>6s}  " + "  ".join(
                f"{vals[k][0]:.3f}±{vals[k][1]:.3f}" for k in ("a_all", "a_sig", "a_noise", "b", "c_all", "c_dead")))
    return rows


if __name__ == "__main__":
    run_vector()
