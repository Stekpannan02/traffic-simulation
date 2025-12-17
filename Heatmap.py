import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random

# ----------------------------------
# Grid settings
# ----------------------------------
GRID_W = 40
GRID_H = 40

# ----------------------------------
# Road mask (EXAKT som din)
# ----------------------------------
road_mask = np.zeros((GRID_H, GRID_W), dtype=int)

for i in range(0, GRID_H, 8):
    road_mask[i, :] = 1
for j in range(0, GRID_W, 8):
    road_mask[:, j] = 1

road_mask[0, :] = 1
road_mask[GRID_H-1, :] = 1
road_mask[:, 0] = 1
road_mask[:, GRID_W-1] = 1

for i in range(1, 8):
    road_mask[i,8] = 0
for i in range(9,32):
    road_mask[i,32] = 0
for j in range(25,39):
    road_mask[16,j] = 0
    road_mask[24,j] = 0
for j in range(9,16):
    road_mask[24,j] = 0
for i in range(9,32):
    road_mask[i,0] = 0
for j in range(0,8):
    road_mask[16,j] = 0
    road_mask[24,j] = 0

# ----------------------------------
# Start & mål (ego-bilen)
# ----------------------------------
start = (0, 16)
goal  = (39, 39)

# ----------------------------------
# Placera "andra bilar"
# ----------------------------------
N_OTHER_CARS = 25
other_cars = []

road_cells = [(r,c) for r in range(GRID_H) for c in range(GRID_W)
              if road_mask[r,c] == 1 and (r,c) not in [start, goal]]

random.seed(3)
other_cars = random.sample(road_cells, N_OTHER_CARS)

# ----------------------------------
# Cost map med trafik
# ----------------------------------
def compute_cost_map():
    cost = np.ones((GRID_H, GRID_W))

    for (r,c) in other_cars:
        cost[r,c] += 5      # lokal kostnad
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc = r+dr, c+dc
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
                cost[nr,nc] += 2  # spill-over

    return cost

# ----------------------------------
# Dijkstra (full heatmap)
# ----------------------------------
def dijkstra(start, goal, cost_map):
    dist = np.full((GRID_H, GRID_W), np.inf)
    prev = {}
    pq = []

    dist[start] = 0
    prev[start] = None
    heappush(pq, (0, start))

    while pq:
        cur_cost, (r,c) = heappop(pq)

        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc = r+dr, c+dc
            if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                continue
            if road_mask[nr,nc] != 1:
                continue

            new_cost = cur_cost + cost_map[nr,nc]
            if new_cost < dist[nr,nc]:
                dist[nr,nc] = new_cost
                prev[(nr,nc)] = (r,c)
                heappush(pq, (new_cost, (nr,nc)))

    return dist, prev

# ----------------------------------
# Reconstruct path
# ----------------------------------
def reconstruct(prev, start, goal):
    path = []
    node = goal
    while node in prev:
        path.append(node)
        node = prev[node]
        if node is None:
            break
    path.reverse()
    return path

# ----------------------------------
# Run
# ----------------------------------
cost_map = compute_cost_map()
dist, prev = dijkstra(start, goal, cost_map)
path = reconstruct(prev, start, goal)

masked_dist = np.ma.masked_where(road_mask == 0, dist)

# ----------------------------------
# Plot
# ----------------------------------
plt.figure(figsize=(8,8))

# Heatmap = bilens tanke
plt.imshow(masked_dist, cmap="inferno")
plt.colorbar(label="Total cost")

# Vägar
plt.imshow(np.ma.masked_where(road_mask == 1, road_mask),
           cmap="gray", alpha=0.35)

# Andra bilar
ys = [r for r,c in other_cars]
xs = [c for r,c in other_cars]
plt.scatter(xs, ys, c="dodgerblue", s=40, label="Other cars")

# Ego-bilens väg
if path:
    px = [p[1] for p in path]
    py = [p[0] for p in path]
    plt.plot(px, py, color="lime", linewidth=3, label="Chosen path")

# Start & mål
plt.scatter(start[1], start[0], c="cyan", s=120, label="Start")
plt.scatter(goal[1], goal[0], c="red", s=120, label="Goal")

plt.title("Dijkstra's algorithm – How the car thinks")
plt.legend()
plt.xticks([]); plt.yticks([])
plt.show()
