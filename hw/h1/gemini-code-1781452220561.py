import random
import copy

# 範例城市座標（X, Y），以便計算距離
CITIES = {
    1: (0, 0),
    2: (1, 5),
    3: (4, 2),
    4: (5, 6),
    5: (2, 2)
}

def calculate_distance(city1, city2):
    """計算兩個城市之間的歐幾里得距離"""
    x1, y1 = CITIES[city1]
    x2, y2 = CITIES[city2]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def height_function(solution):
    """高度函數：旅行推銷員走的總距離 * -1"""
    total_distance = 0
    num_cities = len(solution)
    
    # 計算路徑總距離（包含回到起點）
    for i in range(num_cities):
        current_city = solution[i]
        next_city = solution[(i + 1) % num_cities] # 確保最後一個城市連回第一個城市
        total_distance += calculate_distance(current_city, next_city)
        
    return total_distance * -1

# ==================== 替換成新函式的位置 ====================
def get_neighbors(solution):
    """
    修正後的鄰居函數（2-opt）：
    透過反轉子陣列，完美模擬斷開兩條邊並交叉相連的機制，且包含環狀邊界。
    """
    neighbors = []
    n = len(solution)
    
    # 這裡遍歷所有不相鄰的兩個切點
    for i in range(n):
        for j in range(i + 1, n):
            # 至少要留下兩個城市不被翻轉，且不處理整個陣列完全倒序（等於同一條路徑反向）
            if i == 0 and j == n - 1:
                continue
                
            neighbor = copy.deepcopy(solution)
            # 反轉 i 到 j 之間的部分
            neighbor[i:j+1] = reversed(neighbor[i:j+1])
            neighbors.append(neighbor)
            
    return neighbors
# ==========================================================

def hill_climbing_tsp(num_cities):
    """爬山演算法主流程"""
    # 1. 建立初始解：1 => 2 => 3 => ... => n
    current_solution = list(range(1, num_cities + 1))
    current_height = height_function(current_solution)
    
    print(f"初始解: {current_solution} 走向 1")
    print(f"初始高度 (距離 * -1): {current_height:.2f}\n")
    
    step = 1
    while True:
        neighbors = get_neighbors(current_solution)
        best_neighbor = None
        best_neighbor_height = float('-inf') # 初始設為負無限大
        
        # 評估所有鄰居的高度
        for neighbor in neighbors:
            n_height = height_function(neighbor)
            if n_height > best_neighbor_height:
                best_neighbor_height = n_height
                best_neighbor = neighbor
                
        # 如果最棒的鄰居比當前還要高，就往上爬
        if best_neighbor_height > current_height:
            current_solution = best_neighbor
            current_height = best_neighbor_height
            print(f"步驟 {step}: 找到更好的解 {current_solution}，高度提升至: {current_height:.2f}")
            step += 1
        else:
            # 找不到更高（更短距離）的鄰居，達到局部最佳解（Local Optimum），停止
            print("\n[演算法結束] 已達局部最高峰！")
            break
            
    return current_solution, current_height

# --- 執行程式 ---
if __name__ == "__main__":
    # 假設有 5 個城市
    num_of_cities = len(CITIES)
    best_path, max_height = hill_climbing_tsp(num_of_cities)
    
    print("-" * 30)
    print(f"最終最佳路徑: {' => '.join(map(str, best_path))} => {best_path[0]}")
    print(f"最終最短總距離: {max_height * -1:.2f}")