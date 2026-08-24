"""
Appendix: same FGSM-vs-PGD check as appendix_pgd.py, for the image-domain
CNN (Model A), which is a higher-dimensional, more nonlinear model than the
vector task's MLP -- a place PGD's iterative refinement is more likely to
matter than it did there.
"""

import torch
torch.set_num_threads(1)

from attacks import fgsm_success, pgd_success
from image_stimulus_experiment import (
    make_dataset, SimpleCNN, train, N_TRAIN, N_TEST, EPSILONS, SHAPE_MASK, BG_MASK, SEEDS, N_SEEDS,
)

PGD_STEPS, PGD_RESTARTS = 10, 2
SEEDS = SEEDS[:5]


def ms(vals):
    t = torch.tensor(vals)
    return t.mean().item(), (t.std().item() if len(vals) > 1 else 0.0)


if __name__ == "__main__":
    rows = {eps: {"fgsm": {"all": [], "shape": [], "bg": []}, "pgd": {"all": [], "shape": [], "bg": []}}
            for eps in EPSILONS}

    for s in SEEDS:
        torch.manual_seed(s)
        x_train, y_train = make_dataset(N_TRAIN)
        x_test, y_test = make_dataset(N_TEST)
        model = SimpleCNN()
        train(model, x_train, y_train)

        for eps in EPSILONS:
            rows[eps]["fgsm"]["all"].append(fgsm_success(model, x_test, y_test, eps)[0])
            rows[eps]["fgsm"]["shape"].append(fgsm_success(model, x_test, y_test, eps, mask=SHAPE_MASK)[0])
            rows[eps]["fgsm"]["bg"].append(fgsm_success(model, x_test, y_test, eps, mask=BG_MASK)[0])

            rows[eps]["pgd"]["all"].append(pgd_success(model, x_test, y_test, eps, steps=PGD_STEPS, restarts=PGD_RESTARTS)[0])
            rows[eps]["pgd"]["shape"].append(pgd_success(model, x_test, y_test, eps, mask=SHAPE_MASK, steps=PGD_STEPS, restarts=PGD_RESTARTS)[0])
            rows[eps]["pgd"]["bg"].append(pgd_success(model, x_test, y_test, eps, mask=BG_MASK, steps=PGD_STEPS, restarts=PGD_RESTARTS)[0])
        print(f"seed {s} done", flush=True)

    print(f"Image task, FGSM vs PGD-{PGD_STEPS} ({PGD_RESTARTS} restarts), mean±std over {len(SEEDS)} seeds")
    for eps in EPSILONS:
        for kind in ("fgsm", "pgd"):
            vals = {k: ms(v) for k, v in rows[eps][kind].items()}
            print(f"eps={eps:.2f} {kind:>4s}  all={vals['all'][0]:.3f}±{vals['all'][1]:.3f}  "
                  f"shape={vals['shape'][0]:.3f}±{vals['shape'][1]:.3f}  "
                  f"bg={vals['bg'][0]:.3f}±{vals['bg'][1]:.3f}")
