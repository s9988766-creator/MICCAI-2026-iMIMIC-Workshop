# Archive: early exploratory version (superseded)

These documents record an **early version** of this project, written in June 2026 before the final paper. They are kept for transparency and are **not** the results reported in the paper.

| | Early version (this folder) | Final paper (MLCN 2026) |
|---|---|---|
| Working title | *Do CNNs Learn Radiomic Concepts? A Feature-Space and Attention-Based Audit…* | *Do CNNs Learn Clinically Meaningful Imaging Representations? A Portable Multi-level Representation Audit…* |
| Data split | single 80/10/10 split of the Roboflow redistribution (slice-level; patient IDs unavailable) | official **patient-level 5-fold** split, patient IDs recovered from the original Cheng `.mat` files |
| ResNet-50 accuracy | 99.03% (one test split, n = 308) | **96.46 ± 1.39%** (5-fold, out-of-fold) |
| Attribution | Grad-CAM++ | Grad-CAM with centering, geometric-prior and shifted-input controls |

**The early accuracy numbers are not comparable to the paper.** As the paper reports (Sec. 3.1), a random slice-level split inflates accuracy to about 99% through patient leakage, which is why the final study moved to patient-level cross-validation. Please cite and compare against the numbers in the [paper](https://papers.miccai.org/miccai-2026-sat/paper/MLCN_2026_035.pdf), not the ones in this folder.

Files:

- `EARLY_RESULTS.md`: the original README with the early result tables
- `STRUCTURE.md`, `CLEANUP_PLAN.md`, `RESULTS_REPRODUCE.md`: file layout and reproduction notes for the early pipeline
