# Measuring Geometric Structure in Self-Supervised Representations — Research Plan (ICLR)

## One-line pitch

Use iLab-20M's exact, ground-truth transformation metadata as a precision instrument to characterize what different SSL pretraining objectives actually encode about object *geometry* (pose, scale, location) versus *identity* — turning "SSL learns objectness" from a qualitative, attention-map-based claim into a quantitative, falsifiable measurement across model families, layers, and training design choices.

This is now a self-contained **analysis/understanding paper**, not a detection method paper. No downstream detector, no transfer-to-COCO claim, no box-collapse architecture to design. That removes the biggest risk from the earlier plan (domain-gap on natural-scene transfer) and makes the whole thing tractable in one deadline cycle.

---

## Research question

> **How does the choice of self-supervised pretraining objective — contrastive, self-distillation, masked-image-modeling, or supervised — shape whether and where in the network object geometry (pose, scale, transformation structure) becomes linearly readable, as distinct from object identity/category?**

Framed as a measurement study of *behavior differences between SSL methods*, not a claim about any one method "understanding objects."

---

## Related work — positioning

| Work | Relation |
|---|---|
| **Lenc & Vedaldi 2015**, *Understanding image representations by measuring their equivariance and equivalence* | The core methodology this paper extends — linear equivariance operator + composition test. Originally run on AlexNet under synthetic affine warps; we extend it to a zoo of modern SSL objectives under *real* controlled transformations (pose, background, lighting) with exact ground truth. This is the central novelty claim — an update and broadening of a decade-old measurement technique to the modern SSL landscape, not a new idea from scratch. |
| **Goodfellow, Lee, Le, Saxe, Ng 2009**, *Measuring invariances in deep networks* | Source of the same-object-vs-different-object invariance-ratio test we use, extended with a category-clustering control. |
| **DINO / DINOv2 / iBOT** (self-distillation), **MoCo v3 / SimCLR** (contrastive), **MAE** (masked modeling), **CLIP** (multimodal) | The model zoo — chosen to span the major SSL objective families, not just DINO, so results characterize *objective type*, not one architecture. |
| **"Intriguing Properties of Vision Transformers," "A Cookbook of Self-Supervised Learning," "Do Vision Transformers See Like CNNs?"** | Precedent for this paper's genre: systematic behavioral characterization of representations rather than a new training method. Useful for framing and for reviewer expectations of what counts as a sufficient contribution. |
| **Slot Attention / SAVi** | Cited only as the conceptual source of "object vs. background as a testable representational split," not as a baseline — we are not building an object-centric architecture. |
| **CutLER / SoCo / DetCon / LOST / TokenCut** | Mentioned briefly as *why this measurement matters downstream* (these methods implicitly assume SSL features carry usable geometric structure) — not run as baselines, since we're not proposing or evaluating a competing method. |

---

## Core study (the spine of the paper)

**Data:** iLab-20M. Single object per image, exact metadata (object ID, category, pose/rotation angle, camera azimuth/elevation, scale, background, lighting), used purely as measurement ground truth — never as training signal for the probed models.

**Model zoo** (frozen, no fine-tuning), chosen to span objective families rather than just scale:
- Supervised: ResNet-50, ViT-B/16
- Contrastive: SimCLR-style ResNet, MoCo v3
- Self-distillation: DINO, DINOv2, iBOT
- Masked modeling: MAE
- Multimodal contrastive: CLIP ViT

**Confound to control explicitly:** these models differ in pretraining data scale/diversity, not just objective (e.g. DINOv2 trained on much more/curated data than DINO). Report this as a caveat on any objective-vs-objective ranking, and where feasible use same-data-scale checkpoints or note the confound directly in results rather than papering over it.

**Multi-layer probing:** run every measurement at multiple depths, not just the final embedding — the interesting result is likely *where in the network* geometric structure emerges or disappears, not just whether it exists at the output.

**Transform battery, analyzed separately (do not pool):**
- In-plane 2D transforms (rotation, scale, translation) — a well-behaved group action on pixels.
- Out-of-plane pose/viewpoint change — involves real self-occlusion; expect and characterize partial failure rather than treating it as a simple pass/fail against in-plane results.
- Background swap (same object/pose, different background) — tests object/background separation directly.
- Lighting change — a nuisance-invariance control condition.

**Four measurement families:**

1. **Invariance ratio**, controlled for category clustering: compare `d(z(x), z(T(x)))` against `d(z(x), z(x'))` where `x'` is a *different instance of the same category* — not just a different category, which would trivially reflect known semantic clustering rather than fine-grained identity/pose structure.
2. **Linear equivariance + composition test**, run separately for in-plane and out-of-plane transforms: fit `A_θ` such that `A_θ z(x) ≈ z(T_θ(x))`, then test `A_{θ1} A_{θ2} ≈ A_{θ1+θ2}`. A positive result is evidence the representation approximately linearizes the transformation group; a partial result (holds for small `Δθ`, degrades for large `Δθ` or across occlusion boundaries) is itself an informative, reportable finding.
3. **Spatial object/background correspondence:** for spatial feature maps `F(x)`, compare `sim(F(x)_p, F(T(x))_{T(p)})` for object pixels vs. background pixels — a direct test of whether equivariance itself differs between figure and ground, without needing any box or segmentation head.
4. **Linear probes** on frozen features for object identity, category, rotation, scale, and bounding-box coordinates — per model, per layer. The central comparison: is box-coordinate probing *harder* than identity/category probing, and does that gap shrink or widen with depth, and does it differ by SSL objective family?

**Primary output of the paper:** a systematic map (table/heatmap over model × layer × transform type) of which SSL objectives produce which kind of structure, and where. This is the actual "understanding behavior of SSL methods" contribution.

---

## Controlled interventions (optional depth section, time-permitting)

Turns the Core Study's correlational findings into causal ones by manipulating training recipe and re-running the same diagnostic — this replaces the earlier plan's idea of building a new detector, which is no longer in scope.

- **Augmentation ablation:** vary an SSL training recipe's augmentation set (multi-crop strength, color jitter on/off, whether in-plane geometric transforms are included in the pretext augmentation) and re-run the Core Study's measurements. Tests which specific design choices causally drive (or suppress) emergent geometric structure, rather than only observing correlations across off-the-shelf checkpoints that differ in many confounded ways at once.
- **Targeted auxiliary loss (small-scale only):** add an explicit geometric-equivariance term to one small SSL training run (e.g. a small DINO variant trained on iLab-20M itself) purely to test "does explicitly supervising for this property change what the diagnostic measures" — framed as an ablation probe for the paper's understanding-focused claim, not as a proposed method to be evaluated for downstream detection quality. No box-prediction head, no anti-collapse engineering, no transfer evaluation — that entire apparatus is unnecessary once the goal is measurement rather than a deployable detector.

---

## Risks

- **Scope of claims.** Everything here is measured on turntable single-object images. Be explicit in the paper that results characterize representational behavior *on this controlled distribution*, and do not claim (without evidence) that the same ranking of SSL objectives holds on natural cluttered scenes — that transfer question is explicitly out of scope for this paper, not silently assumed.
- **Confounded model zoo.** Differences in pretraining data scale/curation across DINO/DINOv2/CLIP/MAE etc. can masquerade as differences in "objective." Mitigate by reporting data scale alongside every result and being conservative about causal language in the Core Study (save causal claims for the Controlled Interventions section, where recipe is actually manipulated).
- **Multiple comparisons.** Model × layer × transform-type is a large grid. Use consistent statistical treatment (confidence intervals, corrections where claiming significance) rather than cherry-picking cells that show a clean story.

---

## Scoping against an ICLR deadline

This is now a single, self-contained empirical paper — much more tractable than the earlier three-phase detection plan. The Core Study alone is a complete submission; Controlled Interventions is a nice-to-have depth section if time allows, not a dependency.

**Minimum viable submission:** Core Study only (model zoo × transform battery × four measurement families), framed explicitly as extending Lenc & Vedaldi / Goodfellow et al. to the modern SSL landscape.

**Full submission:** Core Study + Controlled Interventions (augmentation ablation at minimum; the small-scale auxiliary-loss run only if time permits).
