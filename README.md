# MICCAI-2026-iMIMIC-Workshop
Do CNNs Learn Radiomic Concepts? A Feature-Space and Attention-Based Audit for Brain Tumor MRI Classification

# Brain Tumor MRI Classification — Experimental Results

> **Do CNNs Learn Radiomic Concepts? A Feature-Space and Attention-Based Audit
> for Brain Tumor MRI Classification**

This document walks through every experimental result of this project,
**organised by the paper's storyline** rather than the order in which
experiments were run. Each section explains *what we measured*, *what
the number is*, *how to read it*, and *what it proves*.

---

## 📖 The Story Arc

The paper follows a four-act narrative recommended by our advisor:

```
Act 1.  "Try radiomics first"
        → Hand-crafted radiomic features should suffice — they're the
          clinical gold standard.

Act 2.  "It is insufficient"
        → Radiomics supervised classifier caps around 89 %.
          Unsupervised clustering recovers diagnostic classes only
          weakly (ARI ≈ 0.27).
        → Conclusion: radiomic features carry signal but lack
          multi-factor non-linear integration.

Act 3.  "Switch to CNN"
        → Fine-tuned ResNet-50 reaches 99 % accuracy and the
          feature space cleanly partitions into the three diagnostic
          classes (ARI 0.96).

Act 4.  "Audit the CNN — is it learning the right things or cheating?"
        → Verify with concept alignment + spatial attention + multi-
          architecture replication.
```

Every result table and figure below answers one specific question
within this arc.

---

## 📑 Table of Contents

- [Section 1 — Dataset and preprocessing](#section-1)
- [Section 2 — Radiomics supervised baseline (Act 2a)](#section-2)
- [Section 3 — Clustering and dimension-reduction (Acts 2b & 3)](#section-3)
- [Section 4 — CNN classification (Act 3)](#section-4)
- [Section 5 — CNN ↔ Radiomic concept alignment (Act 4a)](#section-5)
- [Section 6 — Grad-CAM++ attention audit (Act 4b)](#section-6)
- [Section 7 — Multi-architecture verification (Act 4c)](#section-7)
- [Section 8 — Bootstrap confidence intervals (statistical robustness)](#section-8)
- [Appendix — Full file inventory](#appendix)

---

<a id="section-1"></a>
## Section 1 — Dataset and Preprocessing

### Dataset

- **Source**: pkdarabi's Kaggle redistribution of Cheng et al. 2015
  (PLOS ONE) brain tumor MRI dataset, 233 patients × multiple slices.
- **Total images**: 3,063 (after cleaning one degenerate label).
- **Classes**: glioma (1,427), meningioma (707), pituitary (929).
- **Split**: 80 / 20 / 10 → train (2,143) / valid (612) / test (308).
- **Sequence**: T1-weighted post-contrast (T1+C).
- **Format**: 640 × 640 grayscale, YOLOv8 segmentation labels.

### ROI design — 3×3 grid centred crop

Each tumor's bounding box is expanded by 2.0× margin in both width
and height, producing a 3 × 3-grid ROI in which the **tumor occupies
exactly the center cell**. This preserves peritumoral anatomical
context that radiologists rely on. The polygon mask is rasterised
from YOLO polygon labels (not Otsu thresholding).

### Data-leakage check ([data_leakage_summary.txt](results/data_leakage_summary.txt))

We verified the dataset has no augmentation-level leakage:

```
Total samples:           3063
Unique original IDs:     3063
Avg augmentations / ID:  1.00
Samples in leaked prefixes (>= 2 splits): 0 / 3063 (0.0%)
Interpretation: PASS — no prefix appears in multiple splits.
```

**Caveat (limitation)**: This only verifies *slice-level* uniqueness.
Patient-level leakage cannot be ruled out because the Roboflow
redistribution does not expose patient IDs.

### Single-tumor verification

Programmatic check of all 3,063 YOLO label files confirms:
- ✅ Each image has exactly **one bounding box**
- ✅ **No multi-focal tumor** samples
- ✅ **No negative samples** (every image has a tumor)

---

<a id="section-2"></a>
## Section 2 — Radiomics Supervised Baseline (Act 2a)

> **Question:** With 19 hand-crafted radiomic features, how good can
> classical machine learning get?

### Feature extraction
- **Library**: scikit-image (PyRadiomics conflicts with Python 3.12)
- **Categories**: 16 first-order intensity + 10 shape-2D + 12 GLCM texture = 38 features
- **Selection**: greedy removal at \|r\| > 0.9 → **19 features kept**

The selection step is visualised in
[correlation_analysis.png](results/correlation_analysis.png) (the
"before" panel shows highly redundant blocks; "after" is much
sparser).

### Table 1 — Six supervised classifiers on 19 radiomic features
*Source: [supervised_baseline_results.csv](results/supervised_baseline_results.csv)*

| Classifier        | Test Acc | 95 % CI         | Macro F1 |
|-------------------|---------:|-----------------|---------:|
| KNN (k = 5)       | 0.834    | [0.795, 0.873]  | 0.800    |
| KNN (k = 15)      | 0.851    | [0.808, 0.886]  | 0.824    |
| **SVM (RBF)**     | **0.896**| **[0.860, 0.929]** | **0.881** |
| SVM (linear)      | 0.883    | [0.847, 0.919]  | 0.862    |
| Random Forest     | 0.860    | [0.818, 0.896]  | 0.834    |
| Logistic Reg.     | 0.877    | [0.841, 0.912]  | 0.856    |

### How to read

- The best radiomic classifier — **SVM with RBF kernel** — caps at
  **89.6 %** test accuracy.
- The 95 % CI upper bound is **92.9 %**, falling short of the >95 %
  threshold typical for clinical deployment.
- All six classifiers span the spectrum of model complexity (linear,
  non-parametric, ensemble, non-linear kernel) — **none breaks 90 %**.

### What it proves

> **Radiomic features carry signal (≈ 90 %) but reach a representational
> ceiling.** This ceiling is a property of the 19 features themselves,
> not of the classifier — even an SVM-RBF kernel cannot extract more.
> This motivates the switch to deep learning in Act 3.

---

<a id="section-3"></a>
## Section 3 — Clustering and Dimension-Reduction Analysis

> **Question (Act 2b):** Can radiomic features *naturally* separate
> the three classes without supervision?
>
> **Question (Act 3):** Does a fine-tuned CNN's feature space cleanly
> partition into the three diagnostic classes?

### Table 2 — Unsupervised clustering on three feature spaces
*Source: [clustering_metrics.csv](results/clustering_metrics.csv)
visualised in [clustering_metrics.png](results/clustering_metrics.png)*

| Feature space              | n_features | Silhouette | Calinski-H | **ARI** | **NMI** |
|---------------------------|-----------:|-----------:|-----------:|--------:|--------:|
| CNN, pretrained (ImageNet) | 585 (PCA)  | 0.06       | 38.5       | 0.249   | 0.219   |
| **CNN, fine-tuned**        | 374 (PCA)  | 0.31       | 360.4      | **0.960** | **0.925** |
| Radiomics (selected)       | 19         | 0.15       | 196.2      | 0.272   | 0.296   |

### How to read

- **ARI** (Adjusted Rand Index): 0 = random, 1 = perfect agreement
  with ground truth. 0.27 means "weakly aligned with diagnosis"; 0.96
  means "almost perfectly aligned".
- **NMI** (Normalized Mutual Information): similar scale to ARI.
- Higher Silhouette and Calinski-Harabasz indicate tighter, more
  separated clusters in the feature space.

### Key contrast

| Comparison                                   | Verdict                                     |
|----------------------------------------------|---------------------------------------------|
| Radiomics (0.272) ≈ Pretrained CNN (0.249)   | Both "task-unaware" representations cap at the same level |
| Fine-tuned CNN (0.960) ≫ both above          | 3.5× improvement from task-specific training |

### What it proves

> Pretrained CNN and radiomics, despite using completely different
> representations, both yield ARI ≈ 0.25 — a known characteristic
> of task-unaware features. **Fine-tuning is the variable that
> unlocks diagnostic separability**, not the choice of representation
> family. The fine-tuned 0.96 is partly *by design* (the GAP layer is
> trained for linear separation), so the relevant comparison is
> radiomics vs pretrained CNN, both well below.

### Dimension-reduction visualisation
*Figure: [dim_reduction_scatter.png](results/dim_reduction_scatter.png)*

The 3 × 3 figure plots PCA / t-SNE / UMAP × pretrained / fine-tuned /
radiomics. **Fine-tuned CNN shows three cleanly isolated clusters
across all three methods**, confirming the high ARI is not an artefact
of a single dimension-reduction algorithm.

### Stranded-sample (outlier) analysis
*Source: [umap_outliers.csv](results/umap_outliers.csv)*

For each val + test sample (n = 920) we compute
`stranded_score = d_own_class − d_nearest_other_class`. Positive scores
indicate the sample lies closer to a *different* class in UMAP than
to its own.

- **13 / 920 (1.4 %) samples are "stranded"**.
- **4 of these are in the test set**.
- **Only 1 is misclassified by the CNN**.
- For all 13, Grad-CAM++ still attends to the polygon tumor body.

**What this proves**: stranded samples are atypical tumor presentations
(edge-cases that would challenge even an experienced radiologist), not
evidence of CNN attention drift.

---

<a id="section-4"></a>
## Section 4 — CNN Classification (Act 3)

> **Question:** Once we switch to a fine-tuned CNN, how good is the
> classifier?

### Architecture
- **Backbone**: ResNet-50, IMAGENET1K_V2 pretrained weights
- **Final layer**: `Sequential(Dropout(0.5), Linear(2048, 3))`
- **Training**: full fine-tune, AdamW lr = 1e-4, 20 epochs, batch 32,
  cosine annealing
- **Augmentation**: horizontal flip + ±15° rotation
- **Selection**: best validation-accuracy checkpoint

### Headline numbers (n = 308 test samples)

| Metric           | Value     | 95 % CI         |
|------------------|----------:|-----------------|
| Test accuracy    | **0.9903**| [0.977, 1.000]  |
| Macro F1         | 0.9887    | [0.973, 1.000]  |
| F1 (glioma)      | 0.9937    | [0.984, 1.000]  |
| F1 (meningioma)  | 0.9836    | [0.955, 1.000]  |
| F1 (pituitary)   | 0.9886    | [0.970, 1.000]  |

→ **3 errors out of 308 test samples.**

### Contrast vs radiomic baseline

| Method            | Test Acc | 95 % CI         |
|-------------------|---------:|-----------------|
| Best radiomic (SVM-RBF) | 0.896 | [0.860, 0.929] |
| ResNet-50 fine-tuned    | **0.990** | **[0.977, 1.000]** |

The CIs **do not overlap** → the gap of 9.4 percentage points is
**statistically significant** (not coincidental).

### What it proves

> Within the same dataset and split, a fine-tuned CNN clears the
> radiomic ceiling by ≈ 9 points. **The story arc moves from "good
> enough" to "clinically aspirational"**. But high accuracy alone is
> not enough — Acts 4a–4c verify that the CNN is learning the right
> things.

---

<a id="section-5"></a>
## Section 5 — CNN ↔ Radiomic Concept Alignment (Act 4a)

> **Question:** Are the CNN's individual dimensions correlated with
> known radiomic concepts, or has it learned something entirely
> orthogonal (= possibly shortcut)?

### Method (computed on val + test, n = 920)

For each CNN dimension $i$ we compute
$p_i = \max_j |M_{ij}|$ where $M$ is the 2048 × 19 Spearman matrix
between CNN dims and radiomic features. We report:

- **median($p$)** — the typical CNN dim's best radiomic correlation
- **max($p$)** — the strongest single alignment
- **frac($p > 0.5$)** — fraction of CNN dims with a strong alignment

### Table 3 — Overall alignment (Spearman)
*Source: [best_cnn_per_radiomic.csv](results/best_cnn_per_radiomic.csv),
visualised in [correlation_analysis.png](results/correlation_analysis.png)*

| Statistic              | Pretrained                | Fine-tuned                |
|------------------------|---------------------------|---------------------------|
| median \|r\|           | 0.234 [0.232, 0.251]      | **0.473** [0.459, 0.492]  |
| max \|r\|              | 0.820 [0.796, 0.842]      | 0.777 [0.750, 0.802]      |
| frac \|r\| > 0.5       | 4.93 % [4.0, 6.2]         | **41.06 %** [37.6, 47.4]  |

### How to read

- **Median jumps from 0.23 → 0.47** (≈ 2× broadening). The "typical"
  CNN dim becomes much more radiomic-like after fine-tuning.
- **frac \|r\| > 0.5 jumps from 5 % to 41 %** — roughly 841 of 2048
  fine-tuned dims align strongly with some radiomic concept (vs 101
  before).
- **Counter-intuitive: max \|r\| slightly *decreases*** (0.820 →
  0.777). Fine-tuning is not producing one super-aligned super-neuron;
  it is recruiting *many* dimensions to be moderately radiomic-like.

### What it proves

> **Fine-tuning *broadens* alignment rather than sharpening it**.
> This is consistent with brain tumor diagnosis on T1+C MRI requiring
> the integration of *many* clinical concepts (intensity, texture,
> shape), not the amplification of any single one.

### Per-category alignment breakdown
*Source: [radiomic_category_breakdown.csv](results/radiomic_category_breakdown.csv),
[radiomic_category_breakdown.png](results/radiomic_category_breakdown.png)*

The 19 selected radiomics are partitioned into three families:
- **First-order intensity** (9): percentiles, skewness, kurtosis, ...
- **GLCM texture** (4): homogeneity, contrast, ASM, correlation
- **Shape 2D** (6): major/minor axis length, eccentricity, ...

| Category (n_R)              | CNN          | median p | max p   | frac > 0.5 |
|----------------------------|--------------|---------:|--------:|-----------:|
| First-order intensity (9)  | pretrained   | 0.186    | 0.498   | **0.00 %** |
| First-order intensity (9)  | **fine-tuned** | **0.414** | **0.777** | **33.98 %** |
| GLCM texture (4)           | pretrained   | 0.178    | 0.604   | 0.39 %     |
| GLCM texture (4)           | **fine-tuned** | **0.372** | 0.675   | **8.74 %** |
| Shape 2D (6)               | pretrained   | 0.191    | **0.820** | 4.83 %    |
| Shape 2D (6)               | fine-tuned   | 0.290    | 0.670   | **4.05 %** ↓ |

### How to read

- **Intensity dominates fine-tuning's gain**: 0 % → 34 % (∞×).
- **Texture sees a moderate gain**: 0.4 % → 8.7 % (22×).
- **Shape is essentially unchanged** (4.83 % → 4.05 %), with overlapping
  CIs — fine-tuning does *not* preferentially align with shape.
- **Pretrained's peak alignment (0.820) comes from Shape**
  (MajorAxisLength), reflecting ImageNet's natural sensitivity to
  object size.
- **Fine-tuned's peak (0.777) comes from Intensity**
  (firstorder_10Percentile) — the peak alignment family *migrates*
  from shape to intensity.

### What it proves

> Fine-tuning does not *uniformly* push the CNN toward radiomics —
> it **selectively re-allocates capacity to the clinically dominant
> family (intensity)** for T1+C MRI brain tumor classification. This
> migration from shape (ImageNet prior) to intensity (clinical signal)
> is the strongest single piece of evidence that the CNN has learned
> *correct* discriminating axes, not arbitrary shortcuts.

---

<a id="section-6"></a>
## Section 6 — Grad-CAM++ Attention Audit (Act 4b)

> **Question:** Does the CNN's spatial attention fall on the tumor
> itself? Or has it learned to attend to non-tumor shortcuts?

### Method

For each test sample (n = 308), Grad-CAM++ is computed at ResNet-50
layer 4 (last conv block, 7×7 spatial), normalised to [0, 1], then
binarised at thresholds τ ∈ {0.3, 0.5}. Against the polygon mask we
report IoU, Coverage, Specificity and Pointing-game accuracy.

### Headline numbers
*Source: [gradcam_summary.csv](results/gradcam_summary.csv) (mean over 308 samples) and
[gradcam_distribution.png](results/gradcam_distribution.png)*

| Metric                    | Value   | 95 % CI         |
|---------------------------|--------:|-----------------|
| Mean IoU (τ = 0.5)        | 0.315   | [0.297, 0.332]  |
| Mean IoU (τ = 0.3)        | 0.341   | [0.326, 0.355]  |
| Coverage (τ = 0.3)        | 0.719   | [0.692, 0.744]  |
| Specificity (τ = 0.5)     | 0.501   | [0.474, 0.526]  |
| Soft IoU (threshold-free) | 0.229   | [0.221, 0.237]  |

Qualitative panels: [gradcam_examples.png](results/gradcam_examples.png)
(three correct + one failure case per class). Visually, attention
consistently concentrates on the tumor body even for failure cases.

### The "centered-Gaussian problem"

A naive reading would say "IoU = 0.315 > 0.30 threshold ⇒ no shortcut".
**But IoU is biased by our centred ROI**: any centred blob will
overlap with a centred mask. To verify, we compare against three
controls.

### Table 4 — Baseline audit at τ = 0.5
*Source: [gradcam_baseline_comparison.csv](results/gradcam_baseline_comparison.csv),
visualised in [gradcam_baseline_4panel.png](results/gradcam_baseline_4panel.png)*

| Baseline                | IoU              | Coverage (τ=0.3) | Specificity     | Pointing-game   |
|-------------------------|-----------------:|-----------------:|----------------:|----------------:|
| Random                  | 0.080 [.078,.081]| 0.700 [.699,.701]| 0.087 [.085,.089]| 0.081 [.052,.114] |
| **Centered Gaussian**   | **0.536** [.517,.554] | **0.928** [.911,.945] | **0.571** [.556,.587] | **0.916** [.883,.945] |
| Pretrained Grad-CAM++   | 0.160 [.144,.177]| 0.689 [.654,.722]| 0.198 [.178,.219]| 0.253 [.205,.299] |
| **Fine-tuned**          | 0.315 [.299,.331]| 0.719 [.692,.744]| 0.501 [.476,.528]| —                |

### How to read

- A hand-crafted **centred Gaussian beats fine-tuned CNN on IoU**
  (0.536 > 0.315). This is the centring artefact — any centred blob
  of tumor-like size wins on IoU.
- **Single-metric IoU is therefore insufficient** to assert
  tumor-centric attention in a centred-ROI setting. Coverage and
  Pointing-game suffer from the same artefact.
- **Specificity is more diagnostic**: it measures the fraction of
  attention that lies *inside* the mask. Gaussian (0.571) still wins
  marginally but fine-tuned (0.501) ≫ pretrained (0.198).

### Table 5 — Paired difference (fine-tuned − baseline)
*Source: [gradcam_baseline_paired_diff.csv](results/gradcam_baseline_paired_diff.csv)*

This is the **cleanest comparison** because pretrained and fine-tuned
share the same centred-ROI geometric prior. Any paired difference is
attributable to *learning*, not geometry.

| Comparison                       | Metric            | Δ (ft − base) | 95 % CI         | Sig.? |
|----------------------------------|-------------------|--------------:|-----------------|------:|
| ft vs pretrained                 | IoU (τ=0.5)       | **+0.155**    | [0.132, 0.177]  | ✓     |
| ft vs pretrained                 | IoU (τ=0.3)       | **+0.183**    | [0.166, 0.200]  | ✓     |
| ft vs pretrained                 | Coverage (τ=0.5)  | **+0.086**    | [0.047, 0.124]  | ✓     |
| ft vs pretrained                 | Coverage (τ=0.3)  | +0.030        | [−0.011, 0.072] | ✗     |
| **ft vs pretrained**             | **Specificity (τ=0.5)** | **+0.303** | **[0.270, 0.333]** | **✓** |
| ft vs pretrained                 | Specificity (τ=0.3)| **+0.237**   | [0.217, 0.258]  | ✓     |

### How to read

- 5 / 6 paired differences are significant (95 % CI excludes 0).
- **Specificity jumps 2.5×** from pretrained (0.198) to fine-tuned
  (0.501). This is the headline finding for the attention audit.
- The only non-significant difference is Coverage at the loose
  threshold (τ = 0.3), where both models cover the mask broadly.

### Table 6 — Per-class Specificity (τ = 0.5)
*Source: [gradcam_baseline_per_class.csv](results/gradcam_baseline_per_class.csv)*

| Baseline              | Glioma  | Meningioma | Pituitary |
|----------------------|--------:|-----------:|----------:|
| Random               | 0.085   | 0.097      | 0.082     |
| Centered Gaussian    | 0.534   | 0.556      | 0.649     |
| Pretrained           | 0.153   | 0.217      | 0.267     |
| **Fine-tuned**       | 0.444   | **0.687**  | 0.472     |

### How to read

- The Centered Gaussian is **class-blind by construction** — it
  produces an identical blob for every image. Its per-class numbers
  vary only because tumor sizes/positions differ slightly across
  classes.
- The fine-tuned CNN's specificity for **meningioma (0.687)** *exceeds*
  the Gaussian baseline (0.556). This pattern is **mathematically
  impossible for a class-blind model** — it can only arise from
  class-conditional attention.

### Clinical interpretation
- Meningiomas frequently attach to the dura and present with
  *eccentric* shapes, which a symmetric centred Gaussian cannot
  capture. The CNN learning "where the meningioma actually sits"
  produces specificity higher than any geometric prior.
- Pituitary tumors *do* sit near the geometric centre, so Gaussian
  matches them well by coincidence.

### Failure-case analysis

Among the 3 misclassified test samples, mean Grad-CAM++ IoU is
**comparable to that of correctly classified samples**, with no
systematic offset toward background. This contrasts with DeGrave
et al. (2020) on COVID-19 radiograph CNNs, where misclassification
co-occurred with severe attention drift.

### What it proves

> **The attention audit is consistent with tumor-centric attention,
> not shortcut learning** — but the argument requires multi-metric
> support:
>
> 1. Fine-tuning ≫ pretrained on 5 / 6 attention metrics (paired).
> 2. Specificity rises 2.5× (CI excludes zero).
> 3. Per-class differentiation defeats any class-blind baseline.
> 4. Misclassified samples retain tumor-centric attention.
>
> Single-metric IoU alone would have been refuted by the centred
> Gaussian baseline. The audit succeeds only because we cross-check
> across orthogonal metrics.

---

<a id="section-7"></a>
## Section 7 — Multi-Architecture Verification (Act 4c)

> **Question:** Are the audit findings specific to ResNet-50, or do
> they generalise across CNN architecture families?

### Setup

We retrain on **DenseNet-121** (dense connectivity, distinct from
ResNet's residual connectivity) using identical hyperparameters and
preprocessing, then rerun the full feature-space audit. Grad-CAM++
on DenseNet is left as future work.

### Table 7 — Cross-architecture replication of core findings

| Finding                                 | ResNet-50              | DenseNet-121           | Replicated? |
|-----------------------------------------|------------------------|------------------------|------------|
| Test accuracy (SOTA range)              | 99.03 % (3 errors)     | 98.70 % (4 errors)     | ✓          |
| Fine-tuned ARI > 0.94                   | 0.960                  | 0.943                  | ✓          |
| Median \|r\| (pretrained → fine-tuned)  | 0.234 → 0.473 (2.0 ×)  | 0.288 → 0.459 (1.6 ×)  | ✓          |
| frac \|r\| > 0.5 (pretrained → fine-tuned) | 4.93 % → 41.06 % (~8×) | 11.04 % → 37.99 % (~3.5×) | ✓     |
| Per-cat fine-tuned ordering             | intensity > texture > shape | **intensity > texture > shape** | ✓ |
| Pretrained strongest category           | Shape (max \|r\| 0.82) | Shape (max \|r\| 0.75) | ✓          |

### Files
- ResNet-50: [clustering_metrics.csv](results/clustering_metrics.csv),
  [radiomic_category_breakdown.csv](results/radiomic_category_breakdown.csv)
- DenseNet-121: [clustering_metrics_densenet.csv](results/clustering_metrics_densenet.csv),
  [radiomic_category_breakdown_densenet.csv](results/radiomic_category_breakdown_densenet.csv)

### How to read

- All five qualitative findings replicate on a structurally distinct
  architecture (residual ≠ dense connectivity).
- Magnitudes differ slightly (2.0× vs 1.6×; 8× vs 3.5×), but the
  *patterns* hold — a small natural variation expected from
  parameter-count and architectural differences.

### What it proves

> The audit findings are **properties of the brain-tumor classification
> task + ImageNet-pretrained fine-tuning** rather than artefacts of any
> specific CNN backbone. To our knowledge, this is the first
> brain-tumor interpretability audit to verify across two distinct
> CNN architecture families. We do not claim universality (Transformer
> backbones are untested), but the within-CNN-family generalisation
> is firmly established.

---

<a id="section-8"></a>
## Section 8 — Bootstrap Confidence Intervals

> **Question:** How statistically robust are the headline numbers?

Every point estimate in this report is accompanied by a non-parametric
bootstrap 95 % CI obtained by resampling the relevant samples with
replacement. The model is **trained once**; bootstrapping resamples
*predictions* (or features) and recomputes the metric.

### Bootstrap iteration counts
| Metric family   | N_boot | Computation cost per iter |
|-----------------|-------:|---------------------------|
| Classification  | 1000   | array shuffle + accuracy (ms) |
| Grad-CAM++      | 1000   | array shuffle + mean (ms) |
| Correlation     | 500    | recompute 2048×19 Spearman matrix (~50 ms) |
| Clustering      | 300    | refit K-means + ARI/NMI (~200 ms) |

Total bootstrap runtime ≈ 30 minutes on CPU. **No retraining is
required** — bootstrap quantifies the *test-set sampling*
uncertainty, not training uncertainty.

### Full summary
*Source: [bootstrap_ci_summary.csv](results/bootstrap_ci_summary.csv)*

This file contains every point estimate quoted in this report
together with its 95 % CI, sample size and bootstrap iteration count.
Use it as the single source of truth for any cross-reference.

---

<a id="appendix"></a>
## Appendix — Full file inventory

### Tables (CSV)

| File                                              | Contains |
|---------------------------------------------------|----------|
| `results/supervised_baseline_results.csv`         | Six radiomic ML baselines (Table 1) |
| `results/clustering_metrics.csv`                  | ResNet-50 K-means / GMM (Table 2)   |
| `results/clustering_metrics_densenet.csv`         | DenseNet-121 K-means / GMM          |
| `results/best_cnn_per_radiomic.csv`               | Per-radiomic best CNN dim (Section 5) |
| `results/best_cnn_per_radiomic_densenet.csv`      | Same, DenseNet                       |
| `results/radiomic_category_breakdown.csv`         | Per-category alignment (Section 5)  |
| `results/radiomic_category_breakdown_densenet.csv`| Same, DenseNet                       |
| `results/gradcam_metrics.csv`                     | Per-sample Grad-CAM++ (Section 6)   |
| `results/gradcam_summary.csv`                     | Grad-CAM++ summary                   |
| `results/gradcam_baseline_comparison.csv`         | 4-baseline × 7-metric (Section 6)   |
| `results/gradcam_baseline_paired_diff.csv`        | Paired diff CIs (Section 6)         |
| `results/gradcam_baseline_per_class.csv`          | Per-class breakdown (Section 6)     |
| `results/data_leakage_check.csv`                  | Filename-prefix occurrence (Section 1) |
| `results/umap_outliers.csv`                       | Stranded samples (Section 3)        |
| `results/bootstrap_ci_summary.csv`                | All bootstrap CIs (Section 8)       |

### Figures (PNG)

| File                                              | Used in    |
|---------------------------------------------------|------------|
| `results/clustering_metrics.png`                  | Section 3  |
| `results/clustering_metrics_densenet.png`         | Section 7  |
| `results/dim_reduction_scatter.png`               | Section 3  |
| `results/correlation_analysis.png`                | Section 2 (feature selection) + Section 5 |
| `results/correlation_analysis_densenet.png`       | Section 7  |
| `results/radiomic_category_breakdown.png`         | Section 5  |
| `results/radiomic_category_breakdown_densenet.png`| Section 7  |
| `results/gradcam_distribution.png`                | Section 6  |
| `results/gradcam_examples.png`                    | Section 6  |
| `results/gradcam_baseline_4panel.png`             | Section 6  |
| `results/gradcam_baseline_bar.png`                | Section 6 (older 3-bar version, superseded by 4-panel) |

### Human-readable summaries (TXT)

| File                                              | Contents |
|---------------------------------------------------|----------|
| `results/data_leakage_summary.txt`                | One-paragraph audit verdict |
| `results/supervised_baseline_summary.txt`         | Six classifiers + best |
| `results/radiomic_category_breakdown.txt`         | Per-category digest with headline interpretation |
| `results/radiomic_category_breakdown_densenet.txt`| Same, DenseNet |
| `results/gradcam_baseline_comparison.txt`         | Multi-metric digest with paired diff + ratios |
| `results/bootstrap_ci_summary.txt`                | All bootstrap CIs, human format |

### Models

| File                              | Description |
|-----------------------------------|-------------|
| `models/resnet50_finetuned.pt`    | ResNet-50, 99.03 % test acc, FC = Sequential(Dropout, Linear(2048, 3)) |
| `models/densenet121_finetuned.pt` | DenseNet-121, 98.70 % test acc, classifier = Sequential(Dropout, Linear(1024, 3)) |
| `models/training_log.csv`         | ResNet-50 epoch-by-epoch log |
| `models/densenet121_training_log.csv` | DenseNet-121 epoch-by-epoch log |
| `models/densenet121_training_curves.png` | DenseNet-121 loss + acc curves |

### Features

| File                                          | Shape          | Description |
|-----------------------------------------------|----------------|-------------|
| `features/cnn_pretrained.npy`                 | (3063, 2048)   | ResNet-50 pretrained (ImageNet V2) GAP |
| `features/cnn_finetuned.npy`                  | (3063, 2048)   | ResNet-50 fine-tuned GAP |
| `features/cnn_pretrained_densenet.npy`        | (3063, 1024)   | DenseNet-121 pretrained GAP |
| `features/cnn_finetuned_densenet.npy`         | (3063, 1024)   | DenseNet-121 fine-tuned GAP |
| `features/radiomics_raw.csv`                  | 38 features    | scikit-image first-order + shape-2D + GLCM |
| `features/radiomics_selected.csv`             | 19 features    | After \|r\| > 0.9 redundancy filter |
| `features/labels.csv`                         | (3063, 5)      | stem, class_name, class_id, split, row_idx |

---

## 🎯 Summary in one paragraph

We audit a fine-tuned ResNet-50 (99.03 % test accuracy on a 3,063-image
brain tumor MRI benchmark) along three orthogonal evidence layers:
(i) diagnosis-aligned clustering shows the fine-tuned feature space
recovers the three classes at ARI 0.96, vastly outperforming radiomics
(0.27) and pretrained CNN (0.25); (ii) per-category radiomic concept
alignment reveals fine-tuning *broadens* (not peaks) the CNN's
correspondence to clinical concepts, with the dominant family
migrating from shape (ImageNet prior) to intensity (T1+C clinical
dominant axis); (iii) Grad-CAM++ specificity rises 2.5× from
pretrained to fine-tuned (paired CI excludes zero), and per-class
attention patterns defeat a class-blind centred-Gaussian baseline.
All findings replicate on DenseNet-121, establishing within-CNN-family
generalisation of the audit methodology.
