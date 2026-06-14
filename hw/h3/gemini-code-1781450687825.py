import random
from nn0 import Value, Adam, linear, cross_entropy

# 1. 準備訓練數據：2維特徵，3個類別 (0, 1, 2)
X = [
    [0.1, 0.2], [0.15, 0.1],  # 類別 0
    [0.8, 0.9], [0.75, 0.85], # 類別 1
    [0.2, 0.8], [0.1, 0.9]    # 類別 2
]
Y = [0, 0, 1, 1, 2, 2]

# 2. 初始化權重矩陣 (形狀為 3 行 2 列：3個輸出類別，2個輸入特徵)
W = [
    [Value(random.uniform(-0.1, 0.1)), Value(random.uniform(-0.1, 0.1))], # 類別 0 的權重
    [Value(random.uniform(-0.1, 0.1)), Value(random.uniform(-0.1, 0.1))], # 類別 1 的權重
    [Value(random.uniform(-0.1, 0.1)), Value(random.uniform(-0.1, 0.1))]  # 類別 2 的權重
]

# 攤平權重送入你的 Adam 優化器
params = [w_ij for w_row in W for w_ij in w_row]
optimizer = Adam(params, lr=0.1)

# 3. 訓練優化迴圈
for epoch in range(100):
    total_loss = Value(0.0)
    
    for x, target in zip(X, Y):
        # 轉成 Value 向量
        x_val = [Value(xi) for xi in x]
        
        # 前向傳播：計算未歸一化的評分 (logits)
        logits = linear(x_val, W)
        
        # 計算交叉熵損失
        loss = cross_entropy(logits, target)
        total_loss += loss
        
    # 平均損失
    mean_loss = total_loss * (1.0 / len(X))
    
    # 反向傳播計算梯度
    mean_loss.backward()
    
    # 執行優化器更新（這會自動清除舊梯度）
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {mean_loss.data:.4f}")

print("\n--- 最終測試預測 ---")
for x, target in zip(X, Y):
    x_val = [Value(xi) for xi in x]
    logits = linear(x_val, W)
    # 挑出數值最大的位置作為預測結果
    pred = max(range(len(logits)), key=lambda i: logits[i].data)
    print(f"輸入: {x} -> 預測類別: {pred} (目標: {target})")