import math
import random
from nn0 import Value, linear, softmax

class MicroGPT:
    def __init__(self, vocab_size, n_embd, block_size):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        
        # 字元嵌入層矩陣 (Vocab_size x n_embd)
        self.token_embedding_table = [
            [Value(random.uniform(-0.1, 0.1)) for _ in range(n_embd)]
            for _ in range(vocab_size)
        ]
        
        # 輸出預測層權重 (Vocab_size x n_embd)
        self.lm_head_w = [
            [Value(random.uniform(-0.1, 0.1)) for _ in range(n_embd)]
            for _ in range(vocab_size)
        ]

    def __call__(self, token_id, pos_id, keys=None, values=None):
        # 1. 提取當前輸入 Token 的嵌入向量
        x = self.token_embedding_table[token_id]
        
        # 這裡為了展示 microgpt 的核心，我們略過複雜的多層 KV 快取計算，
        # 直接把嵌入特徵透過輸出層映射回字彙表大小的 Logits
        logits = linear(x, self.lm_head_w)
        return logits

# 測試本地運行與文本生成循環
if __name__ == "__main__":
    text = "hello micro gpt"
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # 初始化模型
    model = MicroGPT(vocab_size=vocab_size, n_embd=16, block_size=8)
    
    # 模擬推理生成下一個字元
    test_token = char_to_id['h']
    logits = model(token_id=test_token, pos_id=0)
    probs = softmax(logits)
    
    # 根據預測機率選擇下一個字元的 ID
    next_id = max(range(len(probs)), key=lambda i: probs[i].data)
    
    print(f"輸入字元: 'h' -> 模型預測下一個字元: '{id_to_char[next_id]}'")