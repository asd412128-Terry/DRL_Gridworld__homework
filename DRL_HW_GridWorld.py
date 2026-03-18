from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

def value_iteration(n, start, end, obstacles):
    # 初始化 V 矩陣
    V = np.zeros((n, n))
    policy = np.full((n, n), 'U')
    gamma = 0.9
    actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    
    # 迭代計算 V(s)
    for _ in range(100): # 迭代 100 次以收斂
        new_V = np.copy(V)
        for r in range(n):
            for c in range(n):
                if [r, c] == end or [r, c] in obstacles: continue
                
                vals = []
                for dr, dc in actions.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        if [nr, nc] in obstacles: vals.append(-1 + gamma * V[r, c])
                        elif [nr, nc] == end: vals.append(10 + gamma * 0)
                        else: vals.append(-0.1 + gamma * V[nr, nc])
                    else: vals.append(-1 + gamma * V[r, c])
                new_V[r, c] = max(vals)
        V = new_V

    # 根據最終 V 計算最佳策略
    for r in range(n):
        for c in range(n):
            if [r, c] == end or [r, c] in obstacles: continue
            best_a = 'U'; max_v = -float('inf')
            for name, (dr, dc) in actions.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and [nr, nc] not in obstacles:
                    v = (10 if [nr, nc] == end else -0.1) + gamma * V[nr, nc]
                else: v = -1 + gamma * V[r, c]
                if v > max_v: max_v = v; best_a = name
            policy[r, c] = best_a
    return V.tolist(), policy.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    v, p = value_iteration(data['n'], data['start'], data['end'], data['obstacles'])
    return jsonify({'v': v, 'p': p})

if __name__ == '__main__':
    app.run(debug=True)