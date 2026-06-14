import random
from nn0 import Value, linear, softmax, cross_entropy

class MicroRNNLM:
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 1. 輸入到隱藏層的權重矩陣 W_xh (hidden_size x vocab_size)
        self.W_xh = [
            [Value(random.uniform(-0.1, 0.1)) for _ in range(vocab_size)]
            for _ in range(hidden_size)
        ]
        
        # 2. 隱藏層到隱藏層的循環權重矩陣 W_hh (hidden_size x hidden_size)
        # 這是 RNN 記憶歷史資訊的關鍵，Transformer 沒有這種時間步的循環權重
        self.W_hh = [
            [Value(random.uniform(-0.1, 0.1)) for _ in range(hidden_size)]
            for _ in range(hidden_size)
        ]
        
        # 3. 隱藏層到輸出層的權重矩陣 W_hy (vocab_size x hidden_size)
        self.W_hy = [
            [Value(random.uniform(-0.1, 0.1)) for _ in range(hidden_size)]
            for _ in range(vocab_size)
        ]

    def forward(self, input_tokens, target_tokens):
        # 初始化隱藏狀態為全零向量
        h = [Value(0.0) for _ in range(self.hidden_size)]
        total_loss = Value(0.0)
        
        # 依序走過時間步 (Time steps)
        for t in range(len(input_tokens)):
            # 將當前 token 轉換為 One-hot 向量
            x = [Value(0.0) for _ in range(self.vocab_size)]
            x[input_tokens[t]] = Value(1.0)
            
            # RNN 核心公式：h_t = W_xh * x + W_hh * h_{t-1}
            # 這裡為了簡化，先不用 tanh，直接做線性疊加與循環傳遞
            x_part = linear(x, self.W_xh)
            h_part = linear(h, self.W_hh)
            h = [xi + hi for xi, hi in zip(x_part, h_part)]
            
            # 計算當前位置的預測 Logits
            logits = linear(h, self.W_hy)
            
            # 計算與目標 token 之間的交叉熵損失並累加
            loss_t = cross_entropy(logits, target_tokens[t])
            total_loss += loss_t
            
        return total_loss * (1.0 / len(input_tokens))

# 測試運行
if __name__ == "__main__":
    text = "hello"
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    
    # 建立輸入與目標序列：輸入 "hell" -> 預測 "ello"
    inputs = [char_to_id[c] for c in "hell"]
    targets = [char_to_id[c] for c in "ello"]
    
    # 初始化模型
    model = MicroRNNLM(vocab_size=vocab_size, hidden_size=8)
    
    # 前向傳播計算平均損失
    mean_loss = model.forward(inputs, targets)
    print(f"非 Transformer 模型前向傳播成功！當前序列平均 Loss: {mean_loss.data:.4f}")