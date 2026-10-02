# Robust and Non-Robust Features in a Fully Controlled Synthetic Task

Code and paper for a minimal, fully controlled demonstration of the
robust/non-robust feature account of adversarial vulnerability (Ilyas et
al., 2019). We build a synthetic classification task where one input
dimension deterministically decides the label (the "robust" feature) and
five additional dimensions are only probabilistically correlated with it
(the "non-robust" features), plus an image-domain analog (circle vs.
square), and show that:

- a standard MLP/CNN trained on all inputs measurably relies on the
  non-robust features,
- an attacker restricted to perturbing only those features is
  disproportionately effective,
- and removing the non-robust features from the model's input entirely
  (not just zeroing their value) yields a large robustness improvement at
  modest cost to clean accuracy.

The full writeup, including two negative/methodological findings
discovered while building the experiment, is in
[`toy_robust_features_paper.tex`](toy_robust_features_paper.tex) /
[`toy_robust_features_paper.pdf`](toy_robust_features_paper.pdf).

## Layout

**Core experiment**
- [`toy_robust_features_experiment.py`](toy_robust_features_experiment.py) — Models A/B/C on the synthetic vector task; the main result.
- [`attacks.py`](attacks.py) — shared FGSM/PGD implementations used everywhere.
- [`bayes_optimal_analysis.py`](bayes_optimal_analysis.py) — exact Bayes-optimal decision rule for the synthetic task, used as a ceiling on what the noise dimensions can offer.
- [`model_c_robustness.py`](model_c_robustness.py) — checks whether Model C's dead-dimension robustness is fundamental or an initialization artifact.
- [`regularizer_comparison.py`](regularizer_comparison.py) — L1 vs. L2, group Lasso, elastic net, and magnitude pruning at matched clean accuracy.

**Image-domain extension**
- [`image_stimulus_experiment.py`](image_stimulus_experiment.py) — circle-vs-square CNN analog with spatially-restricted attacks.
- [`image_two_branch.py`](image_two_branch.py) — two-branch CNN giving each region its own dedicated weights, the true image-domain analog of Model B.

**Appendix / robustness checks**
- [`appendix_pgd.py`](appendix_pgd.py), [`appendix_pgd_image.py`](appendix_pgd_image.py) — re-run the vector and image tasks under multi-step PGD instead of single-step FGSM.
- [`phase_diagram.py`](phase_diagram.py) / [`phase_diagram_figure.py`](phase_diagram_figure.py) — sweep signal reliability, spurious-feature strength, and weight decay to map where the effect appears vs. vanishes.

**Figures**
- [`make_figures.py`](make_figures.py), [`make_image_figures.py`](make_image_figures.py), [`make_decision_slice_figures.py`](make_decision_slice_figures.py) — regenerate all `fig_*.pdf` outputs used in the paper.

**Root-cause follow-up** ([`root_cause_paper.tex`](root_cause_paper.tex) / [`.pdf`](root_cause_paper.pdf))
- [`root_cause_experiments.py`](root_cause_experiments.py) — E1-E4: Gaussian task with exact Bayes radii vs. trained MLPs (spread over k dimensions, who is vulnerable, interventions).
- [`root_cause_factorial.py`](root_cause_factorial.py) — E5: feature scale vs. reliability.
- [`root_cause_realdata.py`](root_cause_realdata.py) — E6: MNIST with orthogonal block mixing of the input.
- [`root_cause_cifar.py`](root_cause_cifar.py) — E6b: CIFAR-10 CNN restricted to the top-m PCA directions.
- [`root_cause_figures.py`](root_cause_figures.py) — figures and numbers from the `root_cause_*.pkl` results.

## Reproducing

```bash
pip install torch matplotlib numpy
python toy_robust_features_experiment.py   # main vector-task result
python image_stimulus_experiment.py        # image-domain analog
python make_figures.py                     # regenerate figures
```

Each experiment script is self-contained and can be run directly; scripts
under "Appendix" import from the core experiment modules rather than
duplicating model/training code.

## Reference

Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., & Madry, A.
(2019). *Adversarial Examples Are Not Bugs, They Are Features.* NeurIPS.
