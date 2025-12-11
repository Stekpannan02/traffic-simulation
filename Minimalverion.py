import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from heapq import heappush, heappop
import random

# ---------------------------
# 0. Settings
# ---------------------------
GRID_W = 40
GRID_H = 40
SIM_STEPS = 200
N_CARS = 20
p_turn = 0.1  # sannolikhet att svänga på korsning

# ---------------------------
# 1. Road mask
# ---------------------------
road_mask = np.zeros((GRID_H, GRID_W), dtype=int)
for r in range(0, GRID_H, 4):
    if r % 8 == 0:
        road_mask[r, :] = 1
road_mask[GRID_H-1, :] = 1  # sista raden är väg

for c in range(0, GRID_W, 4):
    if c % 8 == 0:
        road_mask[:, c] = 1
road_mask[:, GRID_W-1] = 1  # sista kolumnen är väg

# ---------------------------
# 2. Grid: lista av bilar i varje cell
# ---------------------------
grid = np.empty((GRID_H, GRID_W), dtype=object)
for r in range(GRID_H):
    for c in range(GRID_W):
        grid[r, c] = []

# ---------------------------
# 3. Dijkstra med cellkostnad
# ---------------------------
def dijkstra_path(start, goal, cost_map):
    H, W = GRID_H, GRID_W
    pq = []
    heappush(pq, (0, start))
    visited = {start: None}
    dist = {start: 0}

    while pq:
        cur_cost, cur = heappop(pq)
        if cur == goal:
            break
        r, c = cur
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if 0 <= nr < H and 0 <= nc < W and road_mask[nr, nc] == 1:
                new_cost = cur_cost + cost_map[nr, nc]
                if nxt not in dist or new_cost < dist[nxt]:
                    dist[nxt] = new_cost
                    visited[nxt] = cur
                    heappush(pq, (new_cost, nxt))
    # Backtracking
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = visited.get(node)
    path.reverse()
    return path

# ---------------------------
# 4. Car class
# ---------------------------
class Car:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.pos = start
        self.path = []
        self.index = 0
        self.recalc_path()

    def recalc_path(self):
        cost_map = compute_cost_map()
        self.path = dijkstra_path(self.pos, self.goal, cost_map)
        self.index = 0

    def next_pos(self):
        if self.index + 1 < len(self.path):
            return self.path[self.index + 1]
        return self.pos

# ---------------------------
# 5. Hjälpfunktioner
# ---------------------------
def compute_cost_map():
    cost_map = np.ones((GRID_H, GRID_W))
    for r in range(GRID_H):
        for c in range(GRID_W):
            if len(grid[r, c]) > 0:  # cell upptagen
                cost_map[r, c] = 2
    return cost_map

def is_intersection(pos):
    r, c = pos
    cnt = 0
    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and road_mask[nr, nc] == 1:
            cnt += 1
    return cnt >= 3

# ---------------------------
# 6. Initiera bilar
# ---------------------------
goal = (GRID_H-1, GRID_W//2)  # mitten längst ner

cars = []
for _ in range(N_CARS):
    col = random.randint(0, GRID_W-1)
    start = (0, col)
    car = Car(start, goal)
    cars.append(car)
    grid[start].append(car)

# ---------------------------
# 7. Uppdatera bilar
# ---------------------------
def update_cars():
    cars_to_remove = []
    for car in cars:
        nxt = car.next_pos()
        if nxt != car.pos:
            # Slumpmässig sväng på korsning
            if is_intersection(car.pos) and random.random() < p_turn:
                r, c = car.pos
                candidates = []
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_H and 0 <= nc < GRID_W and road_mask[nr, nc] == 1:
                        if (nr, nc) != car.pos and len(grid[nr, nc]) == 0:
                            candidates.append((nr, nc))
                if candidates:
                    nxt = random.choice(candidates)
                    car.pos = nxt
                    # låt nästa recalc ske vid nästa korsning
                    car.index = 0
                    nxt = car.next_pos()

            # Flytta i grid
            old_r, old_c = car.pos
            if car in grid[old_r, old_c]:
                grid[old_r, old_c].remove(car)
            nr, nc = nxt
            grid[nr, nc].append(car)
            car.pos = nxt
            car.index += 1

        # Kolla mål
        if car.pos == car.goal:
            if car in grid[car.pos[0], car.pos[1]]:
                grid[car.pos[0], car.pos[1]].remove(car)
            cars_to_remove.append(car)

    for car in cars_to_remove:
        cars.remove(car)

# ---------------------------
# 8. Visualization
# ---------------------------
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(road_mask, cmap="gray_r")
ax.set_xticks([]); ax.set_yticks([])

dots = []
for car in cars:
    dot, = ax.plot([car.pos[1]], [car.pos[0]], 'o', color='red', markersize=5)
    dots.append(dot)

def animate(step):
    update_cars()
    while len(dots) > len(cars):
        dot = dots.pop()
        dot.remove()
    while len(dots) < len(cars):
        car = cars[len(dots)]
        dot, = ax.plot([car.pos[1]], [car.pos[0]], 'o', color='red', markersize=5)
        dots.append(dot)
    for dot, car in zip(dots, cars):
        dot.set_data([car.pos[1]], [car.pos[0]])
    ax.set_title(f"Step {step} | Cars: {len(cars)}")
    return dots

anim = FuncAnimation(fig, animate, frames=SIM_STEPS, interval=80, blit=False)
plt.show()
