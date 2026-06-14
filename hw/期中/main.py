import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 根據你的 nn0.py 實際導出內容進行修正
from nn0 import Value, adam, matmul, add, softmax, log

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df.iloc[:, [0, 1]]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df['text'].tolist(), df['label'].tolist()

def main():
    print("========== 1. 資料載入與特徵工程 ==========")
    texts, labels = load_and_preprocess_data('spam.csv')
    
    # 使用 TF-IDF 轉成數位矩陣（特徵數限制為 100，避免底層鏈結過載）
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(texts).toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    print(f"訓練集樣本數: {len(y_train)} | 測試集樣本數: {len(y_test)}")

    print("\n========== 2. 機器學習：隨機森林模型 ==========")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    rf_preds = rf_model.predict(X_test)
    print(f"隨機森林測試集準確率: {accuracy_score(y_test, rf_preds) * 100:.2f}%")

    print("\n========== 3. 底層驗證：自訂 nn0 引擎微型訓練 ==========")
    # 抽取 20 筆樣本進行底層訓練 PoC
    X_poc = X_train[:20]
    y_poc = y_train[:20]
    
    # 初始化權重 W (大小為 100 x 2) 與 偏置 b (大小為 1 x 2)
    W = [[Value(random.uniform(-0.01, 0.01)) for _ in range(2)] for _ in range(100)]
    b = [[Value(0.0), Value(0.0)]]
    
    # 收集需要優化的參數列表
    params = [w_ij for row in W for w_ij in row] + b[0]
    
    # 初始化你的 adam 優化器狀態狀態字典
    m = [0.0] * len(params)
    v = [0.0] * len(params)
    
    print("使用 nn0.py 的算子與 adam 函式優化中...")
    for epoch in range(5):
        epoch_loss = Value(0.0)
        
        for x_vec, target in zip(X_poc, y_poc):
            # 將輸入轉為 1 x 100 的二維列表（滿足你的 matmul 矩陣形狀要求）
            x_val = [[Value(float(xi)) for xi in x_vec]]
            
            # 前向傳播：Outputs = matmul(X, W) + b
            xw = matmul(x_val, W)
            logits = add(xw, b)[0]  # 取出第一列
            
            # 計算 Softmax 預測機率
            probs = softmax(logits)
            
            # 計算交叉熵損失：Loss = -log(probs[target])
            # 利用你的 log 算子與內建的運算子重載
            loss = Value(0.0) - log(probs[target])
            epoch_loss = epoch_loss + loss
            
        # 計算平均損失
        mean_loss = epoch_loss * (1.0 / len(X_poc))
        
        # 歸零歷史梯度 (手動重置 params 的 grad)
        for p in params:
            p.grad = 0.0
            
        # 反向傳播
        mean_loss.backward()
        
        # 調用你 nn0.py 裡面的 adam 函式更新參數
        adam(params, m, v, lr=0.01, t=epoch+1)
        
        print(f"Epoch {epoch+1} | 損失 Loss: {mean_loss.data:.4f}")

if __name__ == "__main__":
    main()