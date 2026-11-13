<div align="center">

# 乳房攝影腫塊良惡性分類：<br>手工形狀特徵 + KNN 與遷移學習 CNN 的比較

**賴鉑崴（Bo-Wei Lai）**

國立陽明交通大學 · 機器學習演算法與應用 期末專題（2026 春季，李昕潔老師）

[![Data](https://img.shields.io/badge/Data-CBIS--DDSM-6f42c1.svg)](https://www.kaggle.com/datasets/dynemiesizumaki/cbis-ddsmpng)
[![Framework](https://img.shields.io/badge/PyTorch-scikit--learn-ee4c2c.svg)](#環境與重現)
[![Period](https://img.shields.io/badge/2026-Feb%20%E2%80%93%20Jun-1f6feb.svg)](#)

</div>

---

## 一句話重點

在 CBIS-DDSM 乳房攝影上比較兩條路線：**傳統機器學習**（從腫塊遮罩抽 8 個形狀特徵，再用 KNN 分類）與**深度學習**（ImageNet / RadImageNet 預訓練 CNN 微調）。在病人層級切分下，最好的設定是 **ResNet18 + 病灶 bounding-box 裁切**，測試集 AUC 0.77、準確率 0.72。

| | 發現 | 數字（測試集） |
|---|---|---|
| 1 | 從原圖依病灶 bounding box 裁切（含周圍組織）優於直接使用 ROI 小圖 | AUC **0.77** vs 0.73 |
| 2 | 醫學影像預訓練（RadImageNet）沒有比 ImageNet 好 | AUC 0.67 vs 0.73 |
| 3 | 最佳模型的敏感度與特異度 | 惡性召回 0.62、良性召回 0.79 |
| 4 | 影像層級切分的兩個實驗無法和病人層級的結果直接比較 | 見 [限制](#限制) |

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

1. **特徵抽取**（`KNN pipeline/01_feature_extraction.py`）：從遮罩取最大連通區域，計算 8 個形狀特徵：面積、周長、圓度、離心率、堅實度（solidity）、長寬比、邊界粗糙度、周長面積比
2. **分類**（`KNN pipeline/02_knn_classifier.py`）：
   - 前向特徵選擇（`SequentialFeatureSelector`）＋ k = 3, 5, …, 15 的 5-fold 交叉驗證
   - 最後選定 **k = 11**
   - 漏診癌症的代價高，因此另外把惡性判斷門檻從 0.5 降到 **0.4**，並以惡性召回率作為選擇標準

### 路線二：遷移學習 CNN

共通設定：AdamW、加權交叉熵處理類別不平衡、翻轉 / 旋轉 / 色彩擾動資料增強、以驗證集 AUC 選最佳 checkpoint。

| # | Notebook | 模型（預訓練） | 影像策略 | 切分 |
|---|---|---|---|---|
| 01 | `01_resnet50_baseline.ipynb` | ResNet50（ImageNet），凍結大部分 backbone | 直接使用 `roi_cropped.png` | 病人層級 |
| 02 | `02_resnet18_bbox_crop.ipynb` | ResNet18（ImageNet） | 由遮罩求 bounding box，外擴 30% 後從原圖裁切；推論時 5 次 TTA | 病人層級 |
| 03 | `03_resnet50_radimagenet.ipynb` | ResNet50（RadImageNet），全部微調 | 同 02 的 bounding-box 裁切 | 病人層級 |
| 04 | `04_resnet50_padding_preprocess.ipynb` | ResNet50（ImageNet） | ROI 小圖補黑邊維持長寬比 | 影像層級 |
| 05 | `05_densenet121_baseline.ipynb` | DenseNet121（ImageNet） | 同 04 | 影像層級 |

## 結果

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

**KNN（路線一）**：*待補：最佳 k = 11 下的交叉驗證準確率，以及門檻 0.4 時的惡性召回率。*

### 觀察

1. **保留病灶周圍組織有幫助**：從原圖依 bounding box 外擴 30% 裁切（02）比只用緊貼病灶的 ROI 小圖（01）AUC 高，推測周圍組織的結構扭曲等線索對判斷良惡性有用。
2. **醫學影像預訓練不一定比較好**：RadImageNet 以 CT、MRI、超音波為主，乳房攝影佔比很低；在這麼小的資料集上，ImageNet 的通用低階特徵（邊緣、紋理）仍有競爭力。
3. **準確率不是唯一指標**：01 的惡性召回率（0.74）其實比 02（0.62）高。若臨床上更在意不漏掉癌症，可能要選 01，或對 02 降低判斷門檻。

## 限制

- **測試集很小**（約 71 例）：AUC 0.77 這樣的數字，95% 信賴區間大約有 ±0.1，實驗之間 0.04 左右的差距不一定顯著。
- **各實驗的案例集合不同**：01 用 495 例，02、03 因遮罩條件排除部分案例只剩 458 例，測試集也不同，所以比較只能算粗略參考。
- **04、05 為影像層級切分**：同一位病人的影像可能同時出現在訓練和測試集，結果會被高估，不能和 01–03 直接比較。
- 每個設定只跑一次，沒有重複不同隨機種子。

## 環境與重現

```bash
pip install torch torchvision scikit-learn scikit-image opencv-python pandas matplotlib pillow kaggle
# 將 kaggle.json（https://www.kaggle.com/settings）放到 ~/.kaggle/
kaggle datasets download -d dynemiesizumaki/cbis-ddsmpng --unzip -p ./data
```

CNN notebook 在 Google Colab T4 GPU 上執行，單一 notebook 約 15–30 分鐘。

```
KNN pipeline/
  01_feature_extraction.py     從遮罩抽取 8 個形狀特徵 → features0512.csv
  02_knn_classifier.py         特徵選擇、k 值搜尋、惡性門檻調整
CNN pipeline/
  01–05_*.ipynb                五個 CNN 實驗（含輸出結果）
```
