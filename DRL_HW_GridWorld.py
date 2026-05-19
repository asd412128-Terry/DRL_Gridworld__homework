from flask import Flask, render_template, request, jsonify
import numpy as np
import random

app = Flask(__name__)

# 定義全域環境變數
GAMMA = 0.9
ACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

def get_reward_and_next_state(r, c, dr, dc, n, end, obstacles):
    """計算給定動作後的 Reward 與下一個 State"""
    nr, nc = r + dr, c + dc
    # 檢查是否在邊界內
    if 0 <= nr < n and 0 <= nc < n:
        if [nr, nc] in obstacles:
            return -1, r, c  # 撞到障礙物，給予懲罰並停在原地
        elif [nr, nc] == end:
            return 10, nr, nc  # 抵達終點，給予高獎勵
        else:
            return -0.1, nr, nc # 正常移動，給予微小步數懲罰
    else:
        return -1, r, c # 撞到牆壁(出界)，給予懲罰並停在原地

def policy_evaluation(n, end, obstacles, policy):
    """HW1-2: 策略評估 (Policy Evaluation) - 計算給定策略下的價值"""
    V = np.zeros((n, n))
    for _ in range(100): # 迭代 100 次以收斂
        new_V = np.copy(V)
        for r in range(n):
            for c in range(n):
                if [r, c] == end or [r, c] in obstacles: 
                    continue
                
                # 直接取出該格子目前策略指定的動作，不使用 max()
                action = policy[r][c]
                dr, dc = ACTIONS[action]
                reward, nr, nc = get_reward_and_next_state(r, c, dr, dc, n, end, obstacles)
                
                # Bellman Equation (沒有 max)
                new_V[r, c] = reward + GAMMA * V[nr, nc]
        V = new_V
    return V.tolist()

def value_iteration(n, end, obstacles):
    """HW1-3: 價值迭代 (Value Iteration) - 尋找最佳價值與最佳策略"""
    V = np.zeros((n, n))
    policy = np.full((n, n), 'U', dtype=str)
    
    # 1. 迭代計算最佳 V(s)
    for _ in range(100): 
        new_V = np.copy(V)
        for r in range(n):
            for c in range(n):
                if [r, c] == end or [r, c] in obstacles: 
                    continue
                
                vals = []
                for action, (dr, dc) in ACTIONS.items():
                    reward, nr, nc = get_reward_and_next_state(r, c, dr, dc, n, end, obstacles)
                    vals.append(reward + GAMMA * V[nr, nc])
                # Bellman Optimality Equation (取 max)
                new_V[r, c] = max(vals)
        V = new_V

    # 2. 根據最終 V 計算最佳策略 (Greedy Policy)
    for r in range(n):
        for c in range(n):
            if [r, c] == end or [r, c] in obstacles: 
                continue
            
            best_a = 'U'
            max_v = -float('inf')
            for action, (dr, dc) in ACTIONS.items():
                reward, nr, nc = get_reward_and_next_state(r, c, dr, dc, n, end, obstacles)
                v = reward + GAMMA * V[nr, nc]
                if v > max_v: 
                    max_v = v
                    best_a = action
            policy[r, c] = best_a
            
    return V.tolist(), policy.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hw1_2', methods=['POST'])
def hw1_2():
    data = request.json
    n, end, obstacles = data['n'], data['end'], data['obstacles']
    
    # 1. 隨機生成行動 (Random Policy)
    policy = np.full((n, n), 'U', dtype=str)
    action_keys = list(ACTIONS.keys())
    for r in range(n):
        for c in range(n):
            if [r, c] == end or [r, c] in obstacles: 
                continue
            policy[r, c] = random.choice(action_keys)
            
    # 2. 計算該隨機策略的價值評估
    V = policy_evaluation(n, end, obstacles, policy.tolist())
    return jsonify({'v': V, 'p': policy.tolist()})

@app.route('/hw1_3', methods=['POST'])
def hw1_3():
    data = request.json
    # 執行價值迭代，得出最佳 V 與最佳 Policy
    v, p = value_iteration(data['n'], data['end'], data['obstacles'])
    return jsonify({'v': v, 'p': p})

if __name__ == '__main__':
    app.run(debug=True)
