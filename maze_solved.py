import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from matplotlib.animation import FuncAnimation

# 迷宮大小（行數和列數）
n, m = 50, 50

# 初始化迷宮，1代表牆，0代表路徑
maze = np.ones((n, m))

# 定義四個方向的移動
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def generate_maze(x, y):
    maze[x][y] = 0  # 設置起始點為路徑

    random.shuffle(directions)
    for dx, dy in directions:
        nx, ny = x + 2 * dx, y + 2 * dy
        if 0 <= nx < n and 0 <= ny < m and maze[nx][ny] == 1:
            maze[x + dx][y + dy] = 0  # 打通兩點之間的牆
            generate_maze(nx, ny)

# 選擇隨機起始點，並且保證它是奇數
start_x, start_y = random.choice(range(1, n//4, 2)), random.choice(range(1, m//4, 2))

# 隨機選擇一個遠離起點的終點
end_x, end_y = random.choice(range(n*3//4, n, 2)), random.choice(range(m*3//4, m, 2))

generate_maze(start_x, start_y)

# 標記起點，使用不同的值來區分
maze[start_x][start_y] = 2

# 隨機挖空
for i in range(100):
    bx , by = random.choice(range(1, n, 2)), random.choice(range(2, m, 2))
    if bx != start_x and by != start_y:
        maze[bx][by] = 0

# 標記終點
maze[end_x][end_y] = 3

# BFS 狀態初始化
bfs_queue = deque([(start_x, start_y)])
bfs_visited = { (start_x, start_y) }
bfs_parent = { (start_x, start_y): None }
bfs_path = []
bfs_final_path = []

# DFS 狀態初始化
dfs_stack = [(start_x, start_y)]
dfs_visited = { (start_x, start_y) }
dfs_parent = { (start_x, start_y): None }
dfs_path = []
dfs_final_path = []

# 同時運行 BFS 和 DFS 並生成動畫
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

def update(frame):
    for _ in range(10):  # 每一幀處理 10 步
        # BFS 過程
        if bfs_queue:
            bfs_current = bfs_queue.popleft()
            bfs_path.append(bfs_current)
            if bfs_current == (end_x, end_y):
                bfs_queue.clear()
                # BFS 回溯路徑
                step = bfs_current
                steps_count_bfs = 0  # 初始化步數計數器
                while step is not None:
                    bfs_final_path.append(step)
                    step = bfs_parent[step]
                    steps_count_bfs += 1  # 每次回溯增加一步
                print(f"BFS找到的路徑步數: {steps_count_bfs}")
            else:
                x, y = bfs_current
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < m and maze[nx][ny] != 1 and (nx, ny) not in bfs_visited:
                        bfs_queue.append((nx, ny))
                        bfs_visited.add((nx, ny))
                        bfs_parent[(nx, ny)] = (x, y)

        # DFS 過程
        if dfs_stack:
            dfs_current = dfs_stack.pop()
            dfs_path.append(dfs_current)
            if dfs_current == (end_x, end_y):
                dfs_stack.clear()
                # DFS 回溯路徑
                step = dfs_current
                steps_count_dfs = 0  # 初始化步數計數器
                while step is not None:
                    dfs_final_path.append(step)
                    step = dfs_parent[step]
                    steps_count_dfs += 1  # 每次回溯增加一步
                print(f"DFS找到的路徑步數: {steps_count_dfs}")
            else:
                x, y = dfs_current
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < m and maze[nx][ny] != 1 and (nx, ny) not in dfs_visited:
                        dfs_stack.append((nx, ny))
                        dfs_visited.add((nx, ny))
                        dfs_parent[(nx, ny)] = (x, y)

    # 繪製 BFS 圖
    ax1.clear()
    ax1.imshow(maze, cmap='binary')
    ax1.scatter(start_y, start_x, color='red', s=100, label='Start')
    ax1.scatter(end_y, end_x, color='blue', s=100, label='End')

    # 繪製 BFS 的進程
    if bfs_path:
        bfs_path_x, bfs_path_y = zip(*bfs_path)
        ax1.scatter(bfs_path_y, bfs_path_x, color='green', s=50, label='BFS Searching')

    # 畫出 BFS 最後找到的路徑
    if bfs_final_path:
        final_bfs_x, final_bfs_y = zip(*bfs_final_path)
        ax1.plot(final_bfs_y, final_bfs_x, color='red', linewidth=3, label='BFS Path')

    ax1.legend()
    ax1.set_xticks([]), ax1.set_yticks([])
    ax1.set_title("BFS Search")

    # 繪製 DFS 圖
    ax2.clear()
    ax2.imshow(maze, cmap='binary')
    ax2.scatter(start_y, start_x, color='red', s=100, label='Start')
    ax2.scatter(end_y, end_x, color='blue', s=100, label='End')

    # 繪製 DFS 的進程
    if dfs_path:
        dfs_path_x, dfs_path_y = zip(*dfs_path)
        ax2.scatter(dfs_path_y, dfs_path_x, color='yellow', s=50, label='DFS Searching')

    # 畫出 DFS 最後找到的路徑
    if dfs_final_path:
        final_dfs_x, final_dfs_y = zip(*dfs_final_path)
        ax2.plot(final_dfs_y, final_dfs_x, color='blue', linewidth=3, label='DFS Path')

    ax2.legend()
    ax2.set_xticks([]), ax2.set_yticks([])
    ax2.set_title("DFS Search")


ani = FuncAnimation(fig, update, frames=range(1000), repeat=False)
plt.show()
