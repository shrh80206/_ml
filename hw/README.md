# 機器學習期中作業 (第 1 - 6 題完整專案)

本專案包含期中作業第 1 題至第 6 題的完整實作與手算證明。

##  檔案索引與題目對照

### 2. 手算反傳遞演算法
- **對應檔案**：`problem_1_2_gradient.png`
- **說明**：包含 $f(x,y,z)=(x*y)+z$ 與 $f(x,y,z,t)=((x*y)+z)*t$ 的手繪計算圖、正向傳播計算，以及利用鏈鎖規則（Chain Rule）推導梯度的完整手寫過程。

### 3. 基於自訂引擎的分類器訓練
- **對應檔案**：`main.py`, `nn0.py`, `spam.csv`
- **說明**：讀取 `spam.csv` 數據集，使用 `scikit-learn` 的隨機森林（Random Forest）建立垃圾郵件攔截的效能基準（準確率約 97%）；同時調用自研的 `nn0.py` 自動微分算子，進行底層線性分類器的優化驗證。

### 4. 本地 GPT 開發
- **對應檔案**：`microgpt.py`
- **說明**：參考 Andrej Karpathy 的 MicroGPT 架構，實現零外部庫依賴的微型 GPT 訓練與推理核心。

### 6. 非 Transformer 語言模型 
- **對應檔案**：`rnn_lm.py`
- **說明**：不使用 Attention 機制，完全基於傳統循環神經網路（Vanilla RNN）架構與隨時間推移的隱藏狀態（Hidden State），利用純 Python 實現字元級別的語言模型。

## 環境依賴
```bash
pip install pandas scikit-learn
皆參考gemini