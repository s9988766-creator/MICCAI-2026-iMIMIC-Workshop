# Notebook / Repository Cleanup Plan

> 整理 notebook 結構與檔名、**完全不需重訓模型或重跑分析**。
> 所有 `models/*.pt` 與 `features/*.npy` 都不動 — 它們由舊命名產出、
> 改名後仍可被新編號的下游 script 讀取。

---

## 1. Renumbering map

把 `notebooks/` 裡的所有 `.py` 與 `.ipynb` 改名如下。Renumbering 不需修改檔案內容（除了 import 路徑、見 §3）。

| 舊檔名                                         | → 新檔名                                  | 原因 |
|-----------------------------------------------|------------------------------------------|------|
| `00_inspect_dataset.{py,ipynb}`               | `01_inspect_dataset.{py,ipynb}`          | 純編號 |
| `01_preprocess.{py,ipynb}`                    | `02_preprocess.{py,ipynb}`               | 純編號 |
| `01b_data_leakage_check.py`                   | `03_data_leakage_check.py`               | 去掉 `b` |
| `04_extract_radiomics.{py,ipynb}`             | `04_extract_radiomics.{py,ipynb}`        | 不變 |
| `05_feature_selection.{py,ipynb}`             | `05_feature_selection.{py,ipynb}`        | 不變 |
| `05b_supervised_baselines.py`                 | `06_supervised_baselines.py`             | 去掉 `b` |
| `02_train_cnn.{py,ipynb}`                     | `07_train_cnn_resnet50.{py,ipynb}`       | 統一架構命名 |
| `03_extract_cnn_features.{py,ipynb}`          | `08_extract_features_resnet50.{py,ipynb}`| 同上 |
| `06_clustering.{py,ipynb}`                    | `09_clustering.{py,ipynb}`               | 重編號 |
| `07_dim_reduction_viz.{py,ipynb}`             | `10_dim_reduction.{py,ipynb}`            | 重編號 |
| `07b_outlier_analysis.py`                     | `11_outlier_analysis.py`                 | 去掉 `b` |
| `08_correlation_analysis.{py,ipynb}`          | `12_correlation_analysis.{py,ipynb}`     | 重編號 |
| `08b_radiomic_category_breakdown.py`          | `13_per_category_breakdown.py`           | 去掉 `b` |
| `12b_permutation_test_alignment.py` (新)      | `14_permutation_test.py`                 | 移到 alignment 段尾 |
| `09_gradcam_analysis.{py,ipynb}`              | `15_gradcam_analysis.{py,ipynb}`         | 重編號 |
| `09b_save_gradcam_heatmaps.py` (新)           | `16_save_heatmaps.py`                    | Tier 1 add-on |
| `08c_gradcam_baselines.py`                    | `17_gradcam_baselines.py`                | 去掉 `c` |
| `10_bootstrap_ci.py`                          | `18_bootstrap_ci.py`                     | 重編號 |
| `02b_train_cnn_densenet.py`                   | `19_train_cnn_densenet.py`               | 多架構驗證移至最後 |
| `03b_extract_features_densenet.py`            | `20_extract_features_densenet.py`        | 同上 |

最終結構：**01–20 連續編號、無下標 `b/c`、按邏輯流程**。

### Section overview
- **§A (01–03)**: Setup & data validation
- **§B (04–06)**: Radiomics pipeline + supervised baseline
- **§C (07–08)**: ResNet-50 training + feature extraction
- **§D (09–11)**: Unsupervised analysis
- **§E (12–14)**: CNN ↔ radiomic alignment (+ permutation test)
- **§F (15–17)**: Spatial attention audit (+ heatmap save)
- **§G (18)**: Bootstrap CI
- **§H (19–20)**: Multi-architecture verification (DenseNet)

---

## 2. Files to **DELETE** (safe to remove)

These are **historical / redundant** files that no longer serve a purpose
after the cleanup:

### Notebook backups (likely on Desktop)
- `00_inspect_dataset_ipynb_ver_3_0_3x3.ipynb` — 舊版
- `00_inspect_dataset_ipynb_ver_3_1_3x3.ipynb` — 取代成新整理過的 master notebook

### Misleadingly-named files (already moved to `_trash_*` earlier)
- `results/gradcam_baseline_*_densenet.{csv,png,txt}` (5 files) — 內容是 ResNet50 而非 DenseNet
- `results/*_resnet50_safe.{csv,png,npz}` (9 files) — 與主檔名重複

(These were moved to `_trash_20260620_201124/` previously. After verifying nothing was lost, you can `rm -rf _trash_*/` to permanently delete.)

### Optional cleanup
- `features/cnn_*_resnet50.npy` (2 files, 48 MB total) — 與 `cnn_pretrained.npy` / `cnn_finetuned.npy` 完全相同的備份。可刪以省空間；但保留作為「belt-and-suspenders」備份也無害。
- `notebooks/Pipline.txt` — 拼字錯誤 (`Pipline` 應為 `Pipeline`)、且內容已被 RESULTS.md / RESULTS_REPRODUCE.md 取代。可刪。

### **DO NOT DELETE**
- 任何 `models/*.pt` (重訓需 1.5 hr+ GPU)
- 任何 `features/*.npy` (特徵提取需 5 min GPU)
- 任何 `processed/{crops,masks}/` (預處理需 4 min)
- `RESULTS.md` / `RESULTS_REPRODUCE.md` / `STRUCTURE.md` / `CLEANUP_PLAN.md`
- `requirements.txt`
- `paper/main.tex` / `paper/references.bib`

---

## 3. Internal references to update

After renaming, the following references inside the scripts need updating
(grep / sed will catch them):

| 在哪個檔 | 找什麼 | 改成什麼 |
|---------|--------|---------|
| 04, 05, 05b → 06, 07_train | `from .. import 02_train` 之類 | 修為新名稱（若有的話）|
| 08 → 12_correlation | docstring 中 "step 03" 之類 | 改為新編號 |
| 08b → 13_per_category | docstring | 同上 |
| 08c → 17_baselines | docstring 中提到 09 | 改為 15 |
| 09 → 15_gradcam | docstring 中提到 02 model | 改為 07 |
| 09b → 16_save_heatmaps | 同上 | 同上 |
| 10 → 18_bootstrap | docstring 中提到所有 step | 全部更新 |
| 02b → 19_train_densenet | docstring 中提到 02 | 改為 07 |
| 03b → 20_extract_densenet | 同上 | 同上 |

實務做法：renaming 完成後、跑一次 `grep -r 'step 0[2-9]\|step 1[0-2]\|_b\.py\|_c\.py' notebooks/` 找出所有殘留引用。

---

## 4. Renaming command (PowerShell)

在 `C:\Users\s9988\Downloads\brain_tumor_final\notebooks\` 跑一次性 rename。先預覽 (`-WhatIf`)、確認沒問題再正式跑。

```powershell
$rename_map = @{
    '00_inspect_dataset.py'                  = '01_inspect_dataset.py'
    '00_inspect_dataset.ipynb'               = '01_inspect_dataset.ipynb'
    '01_preprocess.py'                       = '02_preprocess.py'
    '01_preprocess.ipynb'                    = '02_preprocess.ipynb'
    '01b_data_leakage_check.py'              = '03_data_leakage_check.py'
    '05b_supervised_baselines.py'            = '06_supervised_baselines.py'
    '02_train_cnn.py'                        = '07_train_cnn_resnet50.py'
    '02_train_cnn.ipynb'                     = '07_train_cnn_resnet50.ipynb'
    '03_extract_cnn_features.py'             = '08_extract_features_resnet50.py'
    '03_extract_cnn_features.ipynb'          = '08_extract_features_resnet50.ipynb'
    '06_clustering.py'                       = '09_clustering.py'
    '06_clustering.ipynb'                    = '09_clustering.ipynb'
    '07_dim_reduction_viz.py'                = '10_dim_reduction.py'
    '07_dim_reduction_viz.ipynb'             = '10_dim_reduction.ipynb'
    '07b_outlier_analysis.py'                = '11_outlier_analysis.py'
    '08_correlation_analysis.py'             = '12_correlation_analysis.py'
    '08_correlation_analysis.ipynb'          = '12_correlation_analysis.ipynb'
    '08b_radiomic_category_breakdown.py'     = '13_per_category_breakdown.py'
    '12b_permutation_test_alignment.py'      = '14_permutation_test.py'
    '09_gradcam_analysis.py'                 = '15_gradcam_analysis.py'
    '09_gradcam_analysis.ipynb'              = '15_gradcam_analysis.ipynb'
    '09b_save_gradcam_heatmaps.py'           = '16_save_heatmaps.py'
    '08c_gradcam_baselines.py'               = '17_gradcam_baselines.py'
    '10_bootstrap_ci.py'                     = '18_bootstrap_ci.py'
    '02b_train_cnn_densenet.py'              = '19_train_cnn_densenet.py'
    '03b_extract_features_densenet.py'       = '20_extract_features_densenet.py'
}

cd 'C:\Users\s9988\Downloads\brain_tumor_final\notebooks\'
foreach ($old in $rename_map.Keys) {
    if (Test-Path $old) {
        # Use -WhatIf first to preview:
        Rename-Item -Path $old -NewName $rename_map[$old] -WhatIf
    }
}
# 確認 preview 正確後、把上一段的 -WhatIf 移除、再跑一次正式 rename。
```

---

## 5. Consolidating the master `.ipynb`

Desktop 上的 `00_inspect_dataset_ipynb_ver_3_1_3x3.ipynb` 是一個 super-notebook，含所有 cells。整理建議：

1. **不要把 21 個 step 全塞同一個 notebook** — 載入時間慢、單一 RAM 容易爆。
2. **改用「3 個 notebook + 1 個 utility module」結構**：

| Notebook | 內含 step | 用途 |
|----------|----------|------|
| `notebook_A_setup_and_radiomics.ipynb` | 01–06 | 資料、預處理、radiomics |
| `notebook_B_cnn_and_audit.ipynb`       | 07–18 | ResNet50 訓練、分析、bootstrap |
| `notebook_C_densenet_replication.ipynb`| 19–20 + downstream rerun | DenseNet 跨架構驗證 |
| `_reproducibility.py` (utility, 不是 notebook) | — | `set_seed`、`seed_worker`、`save_to_drive` |

3. **每個 notebook 最上面有「Setup cell」**：
   - mount Drive
   - `import _reproducibility; set_seed(42)`
   - 定義 `BASE`, `RESULTS_DIR` 等共用路徑

---

## 6. Suggested .gitignore (for GitHub)

```gitignore
# Datasets & raw downloads
dataset/
*.zip
*.tar.gz
inspect_metadata.csv

# Trained models (too large for GitHub free tier)
models/*.pt
models/*.pth

# Extracted features (large binary)
features/*.npy

# Generated crops & masks (large; regeneratable from raw data)
processed/crops/
processed/masks/

# Heatmaps (regeneratable from saved model)
results/gradcam_heatmaps/

# Trash folders from cleanup
_trash_*/

# Python cache
__pycache__/
*.pyc
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
```

**要 push 上 GitHub 的內容**：
- ✅ All `.py` and `.ipynb` notebooks
- ✅ `RESULTS.md`, `RESULTS_REPRODUCE.md`, `STRUCTURE.md`, `CLEANUP_PLAN.md`
- ✅ `requirements.txt`, `.gitignore`
- ✅ `results/*.csv`, `results/*.png`, `results/*.txt` (all small, all valuable)
- ✅ `paper/main.tex`, `paper/references.bib`
- ❌ Models, large features, raw dataset (regeneratable; see `.gitignore`)

GitHub free tier 上限 100 MB / file、整 repo 1 GB；上述策略可確保 well under 100 MB total。

---

## 7. Recommended order to execute the cleanup

| 順序 | 行動 | 時間 | 風險 |
|------|------|------|------|
| 1 | Backup 整個 `brain_tumor_final/` 到另一個位置 | 5 min | 0 (純複製) |
| 2 | 用 PowerShell `Rename-Item -WhatIf` 預覽改名 | 1 min | 0 |
| 3 | 確認預覽無誤後、正式跑 rename | 30 s | 低 |
| 4 | `grep -r` 找內部殘留引用、改 docstring | 10 min | 低 |
| 5 | 跑 `01_inspect_dataset.py` 確認讀寫路徑正確 | 1 min | 低 |
| 6 | (可選) 重組成 3 個 notebook | 30 min | 中 |
| 7 | 寫 `.gitignore` + push 上 GitHub | 10 min | 0 |

**整個清理過程 ~1 小時、完全不需 GPU、完全不影響任何數據**。

---

## 8. After cleanup — what your repo looks like

```
brain_tumor_final/
├── README.md                    ← GitHub 首頁（指向 RESULTS.md）
├── RESULTS.md                   ← 完整 results 講解
├── RESULTS_REPRODUCE.md         ← 如何重現
├── STRUCTURE.md                 ← 檔案結構說明
├── CLEANUP_PLAN.md              ← 本檔（投稿後可刪）
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── _reproducibility.py
│   ├── notebook_A_setup_and_radiomics.ipynb
│   ├── notebook_B_cnn_and_audit.ipynb
│   ├── notebook_C_densenet_replication.ipynb
│   ├── 01_inspect_dataset.py
│   ├── 02_preprocess.py
│   ├── 03_data_leakage_check.py
│   ├── 04_extract_radiomics.py
│   ├── 05_feature_selection.py
│   ├── 06_supervised_baselines.py
│   ├── 07_train_cnn_resnet50.py
│   ├── 08_extract_features_resnet50.py
│   ├── 09_clustering.py
│   ├── 10_dim_reduction.py
│   ├── 11_outlier_analysis.py
│   ├── 12_correlation_analysis.py
│   ├── 13_per_category_breakdown.py
│   ├── 14_permutation_test.py
│   ├── 15_gradcam_analysis.py
│   ├── 16_save_heatmaps.py
│   ├── 17_gradcam_baselines.py
│   ├── 18_bootstrap_ci.py
│   ├── 19_train_cnn_densenet.py
│   └── 20_extract_features_densenet.py
│
├── processed/                   ← 不 push（在 .gitignore）
├── features/                    ← 不 push
├── models/                      ← 不 push
├── results/                     ← push（35 個 csv/png/txt）
└── paper/
    ├── main.tex
    └── references.bib
```

→ **GitHub 上看起來像一個專業的可重現研究 repo**，對推甄 / 履歷加分。
