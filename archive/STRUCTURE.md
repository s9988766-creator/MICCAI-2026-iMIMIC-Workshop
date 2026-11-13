# brain_tumor_final — 檔案結構說明

清理日期：2026/06/20

## 📂 資料夾結構

| 路徑 | 內容 |
|---|---|
| `notebooks/` | 全部 pipeline 程式碼（含 02b / 03b DenseNet 版本） |
| `processed/split.csv` | 預處理後的 80/10/10 split metadata |
| `features/` | CNN GAP 特徵 + Radiomics 特徵 |
| `models/` | 兩個訓練好的模型（ResNet-50 + DenseNet-121） |
| `results/` | 所有分析輸出（兩個架構並列） |

## 🧠 模型檔（models/）

| 檔案 | 大小 | 說明 |
|---|---|---|
| `resnet50_finetuned.pt` | 92 MB | ResNet-50 (2048-D GAP)，test acc **99.03%** |
| `densenet121_finetuned.pt` | 28 MB | DenseNet-121 (1024-D GAP)，test acc **98.70%** |
| `training_log.csv` | — | ResNet-50 訓練曲線 epoch-by-epoch |
| `densenet121_training_log.csv` | — | DenseNet-121 訓練曲線 |
| `densenet121_training_curves.png` | — | DenseNet-121 loss + acc 曲線圖 |

## 📊 特徵檔（features/）

| 檔案 | 維度 | 對應模型 |
|---|---|---|
| `cnn_pretrained.npy` | (3063, 2048) | ResNet-50 pretrained (IMAGENET1K_V2) |
| `cnn_finetuned.npy` | (3063, 2048) | ResNet-50 fine-tuned |
| `cnn_pretrained_resnet50.npy` | (3063, 2048) | (= cnn_pretrained.npy 的備份) |
| `cnn_finetuned_resnet50.npy` | (3063, 2048) | (= cnn_finetuned.npy 的備份) |
| `cnn_pretrained_densenet.npy` | (3063, **1024**) | DenseNet-121 pretrained |
| `cnn_finetuned_densenet.npy` | (3063, **1024**) | DenseNet-121 fine-tuned |
| `radiomics_raw.csv` | 38 features × 3063 samples | scikit-image 自實作的 radiomics |
| `radiomics_selected.csv` | 19 features × 3063 samples | 經 \|r\|>0.9 去冗餘後 |
| `labels.csv` | 5 columns × 3063 samples | stem, class_name, class_id, split, row_idx |

## 📈 分析結果（results/）

### 兩個架構的並列結果

每組分析都有兩個版本：**主檔名 = ResNet-50**、**`_densenet` 後綴 = DenseNet-121**

| 主檔名（ResNet50） | `_densenet` 版本 | 內容 |
|---|---|---|
| `clustering_metrics.csv` | `clustering_metrics_densenet.csv` | k-means / GMM 分群指標 ARI / NMI |
| `clustering_metrics.png` | `clustering_metrics_densenet.png` | 同上之 bar chart |
| `correlation_matrices.npz` | `correlation_matrices_densenet.npz` | Pearson + Spearman 矩陣 |
| `correlation_analysis.png` | `correlation_analysis_densenet.png` | 相關矩陣熱圖 |
| `best_cnn_per_radiomic.csv` | `best_cnn_per_radiomic_densenet.csv` | 每個 radiomic 最強對應 CNN dim |
| `radiomic_category_breakdown.csv` | `radiomic_category_breakdown_densenet.csv` | Per-category alignment + bootstrap CI |
| `radiomic_category_breakdown.png` | `radiomic_category_breakdown_densenet.png` | 同上之 3-panel bar chart |
| `radiomic_category_breakdown.txt` | `radiomic_category_breakdown_densenet.txt` | Human-readable summary |

### ResNet50 專屬結果（**未對 DenseNet 重跑**）

下列分析僅針對 ResNet-50，**沒有 DenseNet 版本**（受限於時間、Grad-CAM++ 需要重跑 09 + 改 08c）：

- `gradcam_metrics.csv` — Grad-CAM++ per-sample IoU / Coverage / Specificity
- `gradcam_summary.csv`、`gradcam_distribution.png`、`gradcam_examples.png` — §3.9 視覺化
- `gradcam_baseline_comparison.csv`、`.txt`、`_per_class.csv`、`_paired_diff.csv` — §3.9.1 baseline 對照
- `gradcam_baseline_4panel.png`、`gradcam_baseline_bar.png` — baseline 視覺化
- `bootstrap_ci_summary.csv`、`.txt` — 10_bootstrap_ci 之全指標 CI
- `data_leakage_check.csv`、`data_leakage_summary.txt` — §2.2.2 資料洩漏檢查
- `dim_reduction_2d.npz`、`dim_reduction_scatter.png` — §3.7 PCA/t-SNE/UMAP
- `umap_outliers.csv` — §3.7.1 stranded sample 分析
- `supervised_baseline_results.csv`、`.txt` — §3.3 KNN/SVM/RF/LR

## 🔍 核心發現對照（給 paper / 老師會議用）

| 指標 | ResNet-50 | DenseNet-121 | 是否重現 |
|---|---|---|---|
| Test accuracy | **99.03%** (3/308 錯) | **98.70%** (4/308 錯) | ✓ 兩者皆 SOTA |
| Fine-tuned ARI | 0.960 | 0.943 | ✓ 跨架構皆 >0.94 |
| Median \|r\| (pre → ft) | 0.234 → 0.473 (**2.0×**) | 0.288 → 0.459 (**1.6×**) | ✓ broadening 重現 |
| Frac>0.5 (pre → ft) | 4.93% → 41.06% (**~8×**) | 11.04% → 37.99% (**~3.5×**) | ✓ 大量 dim 對齊重現 |
| Per-category 排序 (fine-tuned) | intensity > texture > shape | **intensity > texture > shape** | ✓ 完全一致 |
| Pretrained 最強類別 | Shape (max \|r\| 0.820) | **Shape (max \|r\| 0.749)** | ✓ ImageNet 形狀先驗一致 |

→ **5/5 核心發現在 DenseNet 上重現** — paper 的「跨架構通用性」claim 成立。

## ⚠️ 注意事項

1. **Grad-CAM++ 分析僅限 ResNet-50**：DenseNet 版本未跑（需要時間 2-3 hr 額外 GPU）。paper 中應明確標示「Grad-CAM++ 分析限 ResNet-50；DenseNet 注意力分析留作 future work」。

2. **`_trash_*` 子資料夾**：本次清理移除的 14 個誤導 / 重複檔案存放於此，確認沒問題後可永久刪除。

3. **特徵備份檔（`cnn_*_resnet50.npy`）**：與主檔名內容相同，可保留作為「belt-and-suspenders」備份，或刪除以省 48 MB 磁碟空間。
