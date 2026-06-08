# AI CUP 2026 春季賽 — 基於時序資料之桌球戰術與結果預測

**隊伍：** TEAM_10318  
**最終成績：** Private Leaderboard 0.3673 / Rank 35（共 423 隊）

本 repo 為本競賽最終提交（乾淨版，未使用任何評測集洩漏標籤）的完整重現程式碼。
任務為：給定一個回合（rally）的前 *k* 拍時序資料，預測第 *k+1* 拍的
**球種**（actionId，19 類）、**落點**（pointId，10 類）與**該回合發球方是否得分**
（serverGetPoint，二元）。評分 = 0.4×Macro-F1(action) + 0.4×Macro-F1(point) + 0.2×AUC(server)。

---

## 1. 方法概要

系統由三模組組成，最終輸出為兩端機率的「逐任務」加權平均，再經類別遮罩後處理：

1. **序列模型端（NN）** — 五種多任務神經網路集成：Bi-GRU（主力）、Transformer、
   ShuttleNet 啟發版、Bi-LSTM、TCN。共用輸入編碼與四個輸出頭（action / point /
   server / 輔助任務 rally_continue）。
2. **表格模型端（GBDT）** — 68 維手工特徵；boosting（LightGBM / XGBoost / CatBoost）
   + bagging（RandomForest / ExtraTrees，僅 action 任務，針對未見選手泛化）。
3. **後處理** — Position-Aware Class Masking：依桌球物理規則遮去不可能的 action 類別。

兩項關鍵設計：
- **serverGetPoint 洩漏修正**：舊 `test.csv` 洩漏了評測集 67% 回合的 server 答案，
  本版完整遮罩該標籤（`fix_server_leak=True`），僅用其 action/point 序列作為合法擴增資料。
- **未見選手泛化**：測試集 44% 為訓練未見選手，以 bagging 樹針對性補強。

---

## 2. 環境需求

| 項目 | 版本 / 規格 |
|------|------------|
| 平台 | Kaggle Notebook（建議）／ Linux + CUDA GPU |
| GPU  | NVIDIA Tesla **T4 ×2**（16GB×2）。**請勿用 P100**（sm_60 不相容，CUDA 會報 "no kernel image"） |
| Python | 3.10 |
| PyTorch | 2.x（CUDA build） |
| LightGBM | 4.x |
| XGBoost | 2.x |
| CatBoost | 1.x |
| scikit-learn | 1.x（提供 RandomForest / ExtraTrees / GroupKFold） |
| 其他 | numpy、pandas |

於 Kaggle 環境上述套件多為內建。若於本機執行，可安裝：

```bash
pip install torch lightgbm xgboost catboost scikit-learn numpy pandas
```

---

## 3. 資料放置

本 repo **不包含主辦單位資料**（不可重新散布）。請至 AI CUP 競賽頁面下載下列 4 個檔案：

```
train.csv                 # 訓練資料
test_new.csv              # 評測集（最終提交對象）
test.csv                  # 舊測試集（含標籤；僅取 action/point 作合法擴增，server 標籤被遮罩）
sample_submission.csv     # 提交格式範本（用於對齊 rally_uid 與欄位順序）
```

### 路徑設定（重要）

本 notebook 於 **Kaggle Notebook** 開發，資料路徑寫死在 **cell 4（Config）**，預設為
Kaggle dataset 掛載路徑：

```python
train_path     = "/kaggle/input/datasets/kgiprkoio/table-tennis/train.csv"
test_path      = "/kaggle/input/datasets/kgiprkoio/table-tennis/test_new.csv"
sample_path    = "/kaggle/input/datasets/kgiprkoio/table-tennis/sample_submission.csv"
old_test_path  = "/kaggle/input/datasets/kgiprkoio/table-tennis/test.csv"
output_path    = "/kaggle/working/submission_final_v6.csv"
```

- **在 Kaggle 上重現：** 將上述 4 個 csv 掛載為 dataset，使其路徑與 cell 4 一致即可直接執行；
  或修改 cell 4 的路徑對應你掛載的 dataset 位置。
- **在本機 / 其他環境重現：** 請將下載的 4 個 csv 放在任一資料夾，並修改 **cell 4** 的
  `train_path / test_path / sample_path / old_test_path` 指向該資料夾，`output_path` 指向
  可寫入的輸出位置（例如同層 `./submission_final_v6.csv`）。

> **注意：** `test.csv` 的 `serverGetPoint` 為評測答案，程式以 `fix_server_leak=True`（cell 4）
> 在訓練時完整遮罩，請勿關閉此開關。

---

## 4. 執行步驟

最終提交由單一 notebook 重現：

```
桌球_v6.ipynb
```

於 Kaggle Notebook（GPU = T4 ×2）依序執行所有 cell 即可。主要區塊：

1. **套件與設定（Config）** — 超參數與開關（`use_lstm` / `use_tcn` /
   `use_rf_et_action` / `fix_server_leak` 等）。最終提交設定：
   `use_rf_et_action=True、use_lstm=True、use_tcn=True、use_transformer=True、
   use_shuttlenet=True、n_seeds_nn=2、fix_server_leak=True、PRIOR_CORR_SHRINK=0`。
2. **統計量** — 選手 Target Encoding、條件落點、action bigram。
3. **前處理 + Dataset** — 特徵編碼、變長序列 Dataset。
4. **NN 模型定義** — `MultiTaskGRUV3 / TransformerV3 / ShuttleNetV3 / LSTMV3 / TCNV3`。
5. **NN 訓練主迴圈** — `run_nn_track`（5 fold × 2 seed × 5 架構 = 50 模型）+ OOF / test 推論。
6. **GBDT 特徵工程 + 訓練** — `build_tabular_features`（68 維）、
   `fit_gbdt_classifier`（含 lgb/xgb/cat/rf/et 五分支）、task-specific blend。
7. **Ensemble 權重搜尋** — NN 各架構混入權重 + NN↔GBDT 權重 grid search。
8. **Position Masking** — 位置感知類別遮罩。
9. **主流程** — 串接全流程並輸出 `submission_final_v6.csv`。

**總執行時間：** 約 6～7 小時（5 架構 NN 約 4–5h + GBDT 含 RF/ET 約 2h），Kaggle T4 ×2。

---

## 5. 輸入 / 輸出

| | 檔案 | 說明 |
|--|------|------|
| **輸入** | `train.csv`、`test_new.csv`、`test.csv`、`sample_submission.csv` | 主辦資料（需自行下載；路徑於 cell 4 設定，預設為 Kaggle 掛載路徑） |
| **輸出** | `submission_final_v6.csv` | 最終提交檔（每列為 test_new 一個 prefix 的三任務預測；預設輸出至 `/kaggle/working/`） |
| **輸出** | `submission_final_v6_preds.npz` | 模型機率快取（OOF / test 機率，可選；重跑會重新產生） |

提交檔欄位對應評測格式：actionId（19 類機率 / 取 argmax）、pointId（10 類）、
serverGetPoint（機率）。

---

## 6. 重現備註

- 驗證採 **5-Fold GroupKFold（以 match 分組）**，確保同場 rally 不跨 fold，
  所有混合權重與遮罩閾值皆於 OOF 搜尋、不接觸測試集。
- 因含隨機 seed 與 GPU 非決定性運算，重跑分數可能有 ±0.001 量級浮動。
- 程式開發過程使用 Claude（Anthropic）作為程式撰寫輔助工具；
  所有模型設計、特徵工程與實驗分析決策均由參賽者本人完成，未以生成式 AI 產生訓練資料。
