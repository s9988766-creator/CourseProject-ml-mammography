<div align="center">

# 【課程專題】基於 CNN 與形態學特徵提取之乳房腫瘤影像分類研究：<br>深度學習與傳統機器學習之比較

何宜靜、吳昭泓、謝秉岑、黃鈺淇、王彥棻、林鈺昀、林子羽、**賴鉑崴**、張鳳書

國立陽明交通大學 科技管理研究所 · 機器學習演算法與應用 期末專題<br>指導老師：李昕潔（Shin-Jye Lee）· 2026 年 2 月 – 6 月

[![Data](https://img.shields.io/badge/Data-CBIS--DDSM-6f42c1.svg)](https://www.kaggle.com/datasets/dynemiesizumaki/cbis-ddsmpng)

</div>

---

在 CBIS-DDSM 乳房攝影上比較兩條路線：**傳統機器學習**（從腫塊遮罩與影像抽取形態、紋理特徵，再用 KNN 分類）與**深度學習**（ImageNet / RadImageNet 預訓練 CNN 微調）。KNN 經特徵增強、SMOTE 與門檻調整後，惡性召回率由 59.5% 提升到 **83.8%**；CNN 在病人層級切分下最好的是 **ResNet18 + 病灶 bounding-box 裁切**，測試集 AUC **0.77**。

| | 路線 | 發現 | 數字（測試集） |
|---|---|---|---|
| 1 | CNN | 從原圖依病灶 bounding box 裁切（含周圍組織）優於直接使用 ROI 小圖 | AUC **0.77** vs 0.73 |
| 2 | CNN | 醫學影像預訓練（RadImageNet）沒有比 ImageNet 好 | AUC 0.67 vs 0.73 |
| 3 | CNN | 最佳模型的敏感度與特異度 | 惡性召回 0.62、良性召回 0.79 |
| 4 | KNN | 特徵增強（17 → 30 個特徵）對 AUC 幫助最大 | AUC 0.68 → **0.76** |
| 5 | KNN | 門檻由 0.5 降到 0.384，大幅減少漏診 | 惡性召回 59.5% → **83.8%**（漏診 15 → 6 例） |

## 研究問題

乳房攝影中，腫塊的**形狀**（邊界是否規則、是否呈針狀）是放射科判讀良惡性的重要依據。本專題想回答：

1. 只靠遮罩算出的幾個形狀特徵，用簡單的 KNN 能分得多好？
2. 遷移學習的 CNN 能不能直接從影像學到更多資訊？哪些**影像前處理**和**預訓練來源**最重要？

## 資料

- **來源**：[CBIS-DDSM（CLAHE 增強 PNG 版本）](https://www.kaggle.com/datasets/dynemiesizumaki/cbis-ddsmpng)
- **內容**：每個案例有完整乳房攝影（`full_image.png`）、ROI 裁切小圖（`roi_cropped.png`）與病灶遮罩（`roi_mask.png`）
- **案例數**：良性 304、惡性 228（各實驗會排除缺圖或遮罩尺寸不符的案例，實際使用 458–495 例）
- **切分**：實驗 01–03 以 `GroupShuffleSplit` 做**病人層級**切分，同一位病人的影像只會出現在訓練、驗證、測試其中一組（約 70 / 15 / 15）

## 方法

### 路線一：手工形狀特徵 + KNN

1. **特徵抽取**：共 17 個形態與灰階紋理特徵，包括面積、周長、圓度、離心率、堅實度（solidity）、長寬比、邊界粗糙度、周長面積比，以及對比度、能量、熵等紋理特徵。本 repo 的 `knn_handcrafted_features/01_feature_extraction.py` 為其中 8 個形狀特徵的抽取程式
2. **分類**（`knn_handcrafted_features/02_knn_classifier.py`）：
   - **基礎模型**：比較不同 k 值與加權方式，最佳為 k = 3、uniform
   - **三層優化**（漏診癌症代價高，以惡性召回率為主要目標）：
     - 方案 A 特徵增強：加入紋理、形態 × 灰階交互等衍生特徵，17 → 30 個
     - 方案 B SMOTE：在訓練集合成惡性樣本，平衡類別
     - 方案 C 決策門檻：依 Precision–Recall 曲線把惡性門檻由 0.5 調到 0.384
   - **前向特徵選擇 + 複合模型**：以召回率為目標做 `SequentialFeatureSelector`（5-fold），17 個特徵篩到 5 個（周長、邊界粗糙度、對比度、能量、熵），再結合 SMOTE 與門檻調整

### 路線二：遷移學習 CNN

共通設定：AdamW、加權交叉熵處理類別不平衡、翻轉 / 旋轉 / 色彩擾動資料增強、以驗證集 AUC 選最佳 checkpoint。

| # | Notebook | 模型（預訓練） | 影像策略 | 切分 |
|---|---|---|---|---|
| 01 | `01_resnet50_baseline.ipynb` | ResNet50（ImageNet），凍結大部分 backbone | 直接使用 `roi_cropped.png` | 病人層級 |
| 02 | `02_resnet18_bbox_crop.ipynb` | ResNet18（ImageNet） | 由遮罩求 bounding box，外擴 30% 後從原圖裁切；推論時 5 次 TTA | 病人層級 |
| 03 | `03_resnet50_radimagenet.ipynb` | ResNet50（RadImageNet），全部微調 | 同 02 的 bounding-box 裁切 | 病人層級 |
| 04 | `04_resnet50_padding_preprocess.ipynb` | ResNet50（ImageNet） | ROI 小圖補黑邊維持長寬比 | 影像層級 |
| 05 | `05_densenet121_padding_preprocess.ipynb` | DenseNet121（ImageNet） | 同 04 | 影像層級 |

## 結果

### 路線一：KNN（手工形狀特徵）

訓練集 358 例、測試集 90 例（良性 53、惡性 37），80 / 20 分層切分。

| 版本 | 特徵數 | 準確率 | 精確率 | 惡性召回 | F1 | AUC | 混淆矩陣 [[TN FP] [FN TP]] |
|---|---|---|---|---|---|---|---|
| 基礎模型（k = 3） | 17 | 67.8% | 61.1% | 59.5% | 60.3% | 0.68 | [[39 14] [15 22]] |
| A：特徵增強 | 30 | **71.1%** | **64.1%** | 67.6% | 65.8% | **0.76** | [[39 14] [12 25]] |
| B：+ SMOTE | 30 | 67.8% | 59.5% | 67.6% | 63.3% | 0.74 | [[36 17] [12 25]] |
| **C：+ 門檻 0.384** | 30 | 64.4% | 54.4% | **83.8%** | **66.0%** | 0.74 | [[27 26] [6 31]] |
| 複合模型（5 特徵 + SMOTE + 門檻 0.45） | 5 | 53.3% | 46.2% | 81.1% | 58.8% | — | 漏診 7 例 |

- 特徵增強對整體分辨力（AUC）幫助最大；降低門檻則是提升召回率的關鍵，代價是誤診由 14 例增加到 26 例。在篩檢情境中，誤診可由後續超音波或切片確認，這個取捨是合理的。
- 只用前向選擇的 5 個特徵時，測試集召回率掉到 48.6%，需要搭配 SMOTE 與門檻調整才回到 81.1%。

### 路線二：CNN

**病人層級切分**（測試集約 71–73 例）：

| 實驗 | AUC | 準確率 | 敏感度（惡性召回） | 特異度（良性召回） | 混淆矩陣 [[TN FP] [FN TP]] |
|---|---|---|---|---|---|
| 01 ResNet50 + ROI 小圖 | 0.73 | 0.64 | **0.74** | 0.57 | [[24 18] [8 23]] |
| **02 ResNet18 + bbox 裁切** | **0.77** | **0.72** | 0.62 | **0.79** | [[33 9] [11 18]] |
| 03 ResNet50 (RadImageNet) + bbox 裁切 | 0.67 | 0.63 | 0.69 | 0.60 | [[25 17] [9 20]] |

**影像層級切分**（測試集 99 例，數字偏樂觀，僅供參考）：

| 實驗 | 準確率 | 敏感度（惡性召回） | 特異度（良性召回） |
|---|---|---|---|
| 04 ResNet50 + 補黑邊 | 0.68 | 0.50 | 0.81 |
| 05 DenseNet121 + 補黑邊 | 0.61 | 0.55 | 0.65 |

### 觀察

1. **保留病灶周圍組織有幫助**：從原圖依 bounding box 外擴 30% 裁切（02）比只用緊貼病灶的 ROI 小圖（01）AUC 高，推測周圍組織的結構扭曲等線索對判斷良惡性有用。
2. **醫學影像預訓練不一定比較好**：RadImageNet 以 CT、MRI、超音波為主，乳房攝影佔比很低；在這麼小的資料集上，ImageNet 的通用低階特徵（邊緣、紋理）仍有競爭力。
3. **準確率不是唯一指標**：01 的惡性召回率（0.74）其實比 02（0.62）高。若臨床上更在意不漏掉癌症，可能要選 01，或對 02 降低判斷門檻。

## 限制

- **測試集很小**（約 71 例）：AUC 0.77 這樣的數字，95% 信賴區間大約有 ±0.1，實驗之間 0.04 左右的差距不一定顯著。
- **各實驗的案例集合不同**：01 用 495 例，02、03 因遮罩條件排除部分案例只剩 458 例，測試集也不同，所以比較只能算粗略參考。
- **04、05 為影像層級切分**：同一位病人的影像可能同時出現在訓練和測試集，結果會被高估，不能和 01–03 直接比較。
- **KNN 與 CNN 的測試集不同**（KNN 90 例、CNN 約 71 例），兩條路線的數字不能直接逐項比較。
- KNN 的特徵增強與 SMOTE 程式未收錄在本 repo，本 repo 僅含形狀特徵抽取與前向選擇分類程式。
- 每個設定只跑一次，沒有重複不同隨機種子。

## 專案結構

```
CourseProject-ml-mammography/
├── README.md
│
├── knn_handcrafted_features/         # 路線一：傳統機器學習
│   ├── 01_feature_extraction.py      # 從遮罩抽取 8 個形狀特徵
│   └── 02_knn_classifier.py          # KNN 分類 + 特徵選擇 + 門檻調整
│
└── cnn_deep_learning/                # 路線二：深度學習
    ├── 01_resnet50_baseline.ipynb               # ResNet50 + ImageNet 預訓練
    ├── 02_resnet18_bbox_crop.ipynb              # ResNet18 + bbox 裁切
    ├── 03_resnet50_radimagenet.ipynb            # ResNet50 + RadImageNet 預訓練
    ├── 04_resnet50_padding_preprocess.ipynb     # ResNet50 + 補黑邊預處理
    └── 05_densenet121_padding_preprocess.ipynb  # DenseNet121 + 補黑邊預處理
```

## 環境與重現

```bash
pip install torch torchvision scikit-learn scikit-image opencv-python pandas matplotlib pillow kaggle
# 將 kaggle.json（https://www.kaggle.com/settings）放到 ~/.kaggle/
kaggle datasets download -d dynemiesizumaki/cbis-ddsmpng --unzip -p ./data
```

CNN notebook 在 Google Colab T4 GPU 上執行，單一 notebook 約 15–30 分鐘。
