# How to Reproduce All Results in this Repository

> Step-by-step instructions to regenerate every table, figure and CSV
> in [RESULTS.md](RESULTS.md) from a fresh environment.

**Expected total runtime**: ~4.5 hours on a Colab T4 GPU.

---

## 1. Prerequisites

| Component | Required version | Reason |
|-----------|------------------|--------|
| Python    | 3.12             | Tested runtime |
| GPU       | NVIDIA T4 (or comparable; 16 GB VRAM is plenty) | ResNet-50 + DenseNet-121 fine-tune both need GPU |
| Disk      | ~5 GB            | Dataset + crops + features + checkpoints |
| RAM       | 16 GB            | K-means on 920 × 2048-D features |

Install dependencies:

```bash
pip install -r requirements.txt
```

Or on Colab, the first cell of the notebook handles installation.

Kaggle API token (`kaggle.json`) is required to download the dataset.
Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).

---

## 2. Notebook execution order

The pipeline is a 16-step sequence. Each step writes its output to
`processed/`, `features/`, `results/` or `models/`. Later steps read
from those locations. Every step ends with a `save_to_drive()` call to
sync intermediate state to Google Drive (Colab) or to the local
`brain_tumor_final/` mirror (offline).

| # | Script | Reads | Writes | Runtime | Section in RESULTS.md |
|--:|--------|-------|--------|--------:|-----------------------|
| **A — Setup & data** | | | | | |
| 1 | `00_inspect_dataset.py` | Kaggle dataset | `inspect_metadata.csv` | 2 min | §1 |
| 2 | `01_preprocess.py` | dataset | `processed/{crops,masks}/`, `processed/split.csv` | 4 min | §1 |
| 3 | `01b_data_leakage_check.py` | `split.csv` | `results/data_leakage_*` | 30 s | §1 |
| **B — Radiomics pipeline** | | | | | |
| 4 | `04_extract_radiomics.py` | crops, masks | `features/radiomics_raw.csv` (38 feats) | 8 min | §2 |
| 5 | `05_feature_selection.py` | `radiomics_raw.csv` | `features/radiomics_selected.csv` (19 feats) | 30 s | §2 |
| 6 | `05b_supervised_baselines.py` | selected radiomics | `results/supervised_baseline_*` | 1 min | §2 |
| **C — CNN pipeline (ResNet-50)** | | | | | |
| 7 | `02_train_cnn.py` | crops, split | `models/resnet50_finetuned.pt` | ~30 min | §4 |
| 8 | `03_extract_cnn_features.py` | model, crops | `features/cnn_{pretrained,finetuned}.npy` | 3 min | §3-§5 |
| **D — Unsupervised analysis** | | | | | |
| 9 | `06_clustering.py` | CNN + radiomic features | `results/clustering_metrics.{csv,png}` | 1 min | §3 |
| 10 | `07_dim_reduction_viz.py` | same | `results/dim_reduction_*` | 4 min | §3 |
| 11 | `07b_outlier_analysis.py` | dim-reduction output | `results/umap_outliers.csv` | 1 min | §3 |
| **E — CNN ↔ radiomic alignment** | | | | | |
| 12 | `08_correlation_analysis.py` | CNN + radiomic features | `results/correlation_*`, `best_cnn_per_radiomic.csv` | 1 min | §5 |
| 13 | `08b_radiomic_category_breakdown.py` | same | `results/radiomic_category_breakdown.*` | 2 min | §5 |
| 14 | `12b_permutation_test_alignment.py` (Tier 2 add-on) | same | `results/permutation_test_alignment.*` | 3 min | §5 |
| **F — Spatial attention audit** | | | | | |
| 15 | `09_gradcam_analysis.py` | model, crops, masks | `results/gradcam_metrics.csv`, viz | ~10 min | §6 |
| 16 | `09b_save_gradcam_heatmaps.py` (Tier 1 add-on) | model, crops, masks | `results/gradcam_heatmaps/*.npy` + `gradcam_metrics_v2.csv` | ~10 min | §6 |
| 17 | `08c_gradcam_baselines.py` | model, heatmaps, masks | `results/gradcam_baseline_*` | ~12 min | §6 |
| **G — Statistical robustness** | | | | | |
| 18 | `10_bootstrap_ci.py` | everything above | `results/bootstrap_ci_summary.{csv,txt}` | ~30 min | §8 |
| **H — Multi-architecture verification (optional)** | | | | | |
| 19 | `02b_train_cnn_densenet.py` | crops, split | `models/densenet121_finetuned.pt` | ~75 min | §7 |
| 20 | `03b_extract_features_densenet.py` | model, crops | `features/cnn_{pretrained,finetuned}_densenet.npy` | 3 min | §7 |
| 21 | Re-run steps 9, 12, 13 with DenseNet features | — | `*_densenet.{csv,png}` | ~5 min | §7 |

For DenseNet (step 21), point the downstream scripts at the
`_densenet` features. The cleanest pattern is to use a global
`ARCH` variable (suggested in the cleanup plan in §3 below) and have
downstream scripts read `cnn_{pretrained,finetuned}_{ARCH}.npy`.

---

## 3. Reproducibility settings

At the top of *every* training / random-sampling script, call:

```python
from _reproducibility import set_seed, seed_worker, make_generator
set_seed(42)
```

For DataLoader-based scripts (02, 02b, 03, 03b), additionally:

```python
g = make_generator(42)
loader = DataLoader(..., worker_init_fn=seed_worker, generator=g)
```

This enables `cudnn.deterministic` and prevents the worker-process
randomness that defeats `torch.manual_seed` alone. **Trade-off**:
training is ~10–30 % slower but bit-exact reproducible.

---

## 4. Expected key numbers

If you re-run the entire pipeline above, the headline numbers should
land within these tolerances of the published values. The bootstrap
CIs in [RESULTS.md](RESULTS.md) cover this expected variation.

| Metric | Published | Tolerance |
|--------|----------:|----------:|
| ResNet-50 test accuracy | 0.9903 | ± 0.005 |
| ResNet-50 macro F1 | 0.9887 | ± 0.005 |
| Fine-tuned ARI (k-means) | 0.960 | ± 0.020 |
| Fine-tuned median \|r\| | 0.473 | ± 0.020 |
| Fine-tuned frac \|r\|>0.5 | 0.411 | ± 0.030 |
| Grad-CAM++ IoU (τ=0.5) | 0.315 | ± 0.015 |
| Grad-CAM++ specificity (τ=0.5) | 0.501 | ± 0.020 |
| DenseNet-121 test accuracy | 0.9870 | ± 0.005 |
| DenseNet-121 ARI | 0.943 | ± 0.020 |

Tolerances reflect:
- cuDNN non-determinism (even with `deterministic=True`, certain
  reductions are not bit-exact across different GPU SKUs).
- K-means random initialisation (mitigated by `n_init=10`).
- Bootstrap sampling (seed-fixed; exact match expected for the same
  seed).

If your numbers fall outside these tolerances, the most likely causes
are: (a) different `torch` / `cuDNN` version, (b) a different GPU
architecture (T4 vs A100), (c) a missed `set_seed(42)` call, or
(d) different `torchvision` weight enums.

---

## 5. Output verification checklist

After running everything, check that you have:

```
brain_tumor_final/
├── processed/
│   ├── split.csv                              ← 3,063 rows
│   ├── crops/{glioma,meningioma,pituitary}/   ← 3,063 PNGs
│   └── masks/{glioma,meningioma,pituitary}/   ← 3,063 PNGs
├── features/
│   ├── cnn_pretrained.npy                     ← (3063, 2048)
│   ├── cnn_finetuned.npy                      ← (3063, 2048)
│   ├── cnn_pretrained_densenet.npy            ← (3063, 1024)
│   ├── cnn_finetuned_densenet.npy             ← (3063, 1024)
│   ├── radiomics_raw.csv                      ← 38 features
│   ├── radiomics_selected.csv                 ← 19 features
│   └── labels.csv                             ← 3,063 rows
├── models/
│   ├── resnet50_finetuned.pt                  ← ~90 MB, test acc 99.0%
│   └── densenet121_finetuned.pt               ← ~28 MB, test acc 98.7%
└── results/
    ├── (35 result files; see RESULTS.md §Appendix for full inventory)
    └── gradcam_heatmaps/*.npy                  ← 308 heatmaps (~6 MB total)
```

---

## 6. Common issues

| Symptom | Most likely cause | Fix |
|---------|-------------------|-----|
| `kaggle datasets download` fails 401/403 | Missing or invalid `kaggle.json` | Place valid token at `~/.kaggle/kaggle.json` |
| `CUDA out of memory` during 02 / 02b | T4 has 16 GB; batch 32 should fit | Reduce `BATCH_SIZE` to 16, restart runtime |
| `pytorch_grad_cam` not found in 08c / 09 / 09b | Not in default Colab image | First cell installs it; if not, `pip install grad-cam` |
| Re-run accuracy differs by 1–2 % | Different GPU SKU (T4 → A100, etc.) | Expected; bootstrap CIs cover this |
| `features/cnn_*.npy` shape mismatch | Forgot to swap before downstream rerun | Use the `ARCH` variable pattern (see code organisation plan) |
| Drive mount fails | OAuth token expired | Restart Colab runtime + re-mount |

---

## 7. Citing this work

If you build on this repository, please cite (after paper acceptance,
final BibTeX will appear here):

```bibtex
@inproceedings{anonymous2026_brain_tumor_audit,
  title = {Do CNNs Learn Radiomic Concepts? A Feature-Space and
           Attention-Based Audit for Brain Tumor MRI Classification},
  author = {Anonymous Submission},
  booktitle = {MICCAI 2026 Workshop (iMIMIC / MLCN)},
  year = {2026}
}
```
