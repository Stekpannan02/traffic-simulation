
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ------------------------------
# 1. Grid definition
# ------------------------------

GRID_W = 20
GRID_H = 20

# 0 = ingen väg
# 1 = väg
# (vi gör ett enkelt rutnät: var fjärde rad och kolumn är väg)
grid = np.zeros((GRID_H, GRID_W), dtype=int)
for i in range(0, GRID_H, 4):
    grid[i, :] = 1
for j in range(0, GRID_W, 4):
    grid[:, j] = 1

# ------------------------------
# 2. Bilklass
# ------------------------------

class Car:
    def __init__(self, pos, direction):
        self.pos = pos  # (row, col)
        self.direction = direction  # (dr, dc)

    def next_pos(self):
        r, c = self.pos
        dr, dc = self.direction
        return (r + dr, c + dc)

# ------------------------------
# 3. Spawn bilar
# ------------------------------

cars = []

def spawn_car():
    # välj slumpmässig vägcell
    while True:
        r = random.randrange(GRID_H)
        c = random.randrange(GRID_W)
        if grid[r, c] == 1:
            break

    # välj en enkel riktning ±1 cell i en dimension
    direction = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
    return Car((r, c), direction)

# spawn initiala bilar
for _ in range(20):
    cars.append(spawn_car())

# ------------------------------
# 4. Simulationslogik
# ------------------------------

def update():
    occupied = {car.pos for car in cars}

    # försök flytta bilar
    for car in cars:
        nr, nc = car.next_pos()

        # om bilen går utanför grid → byt riktning
        if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
            car.direction = (-car.direction[0], -car.direction[1])
            continue

        # om nästa cell inte är en väg → byt riktning
        if grid[nr, nc] == 0:
            car.direction = (-car.direction[0], -car.direction[1])
            continue

        # om cellen framför är blockerad → stå still
        if (nr, nc) in occupied:
            continue

        # annars flytta bilen
        occupied.remove(car.pos)
        car.pos = (nr, nc)
        occupied.add(car.pos)

# ------------------------------
# 5. Visualisering
# ------------------------------

fig, ax = plt.subplots(figsize=(6,6))

def animate(frame):
    update()

    ax.clear()
    ax.imshow(grid, cmap="gray_r")

    xs = [car.pos[1] for car in cars]
    ys = [car.pos[0] for car in cars]

    ax.scatter(xs, ys, c="red", s=30)
    ax.set_title("Cell-based Traffic Grid")
    ax.set_xticks([])
    ax.set_yticks([])

    return []

anim = FuncAnimation(fig, animate, frames=200, interval=200)
plt.show()

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from heapq import heappush, heappop
import random

# ----------------------------------
# 1. Grid
# ----------------------------------

GRID_W = 20
GRID_H = 20
grid = np.zeros((GRID_H, GRID_W), dtype=int)

# skapa vägar (enkelt korsmönster)
for i in range(0, GRID_H, 4):
    grid[i, :] = 1
for j in range(0, GRID_W, 4):
    grid[:, j] = 1

# start & mål
A = (0, 0)
B = (0, 16)
C = (16, 0)
D = (16, 0)

grid[A] = grid[B] = grid[C] = grid[D] = 1


# ----------------------------------
# 2. Dijkstra med kostnader
# ----------------------------------

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

            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 1:
                new_cost = cur_cost + cost_map[nr, nc]

                if nxt not in dist or new_cost < dist[nxt]:
                    dist[nxt] = new_cost
                    visited[nxt] = cur
                    heappush(pq, (new_cost, nxt))

    # backtracking
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = visited.get(node)

    path.reverse()
    return path


# ----------------------------------
# 3. Bilklass
# ----------------------------------

class Car:
    def __init__(self, start, end):
        self.start = start
        self.goal = end

        self.index = 0  # var i pathen
        self.recalc_delay = random.randint(10, 25)  # frames tills nästa omplanering
        self.path = []
        self.pos = start

    def recalc_path(self, cost_map):
        self.path = dijkstra_path(self.pos, self.goal, cost_map)
        self.index = 0

    def next_pos(self):
        if self.index + 1 < len(self.path):
            return self.path[self.index + 1]
        return self.pos


# ----------------------------------
# 4. Spawn bilar
# ----------------------------------

N_CARS = 40
cars = []

for _ in range(N_CARS):
    start = random.choice([A, B])
    goal = C if start == A else D
    car = Car(start, goal)
    cars.append(car)


# ----------------------------------
# 5. Simulation update
# ----------------------------------

def compute_cost_map(cars):
    cost = np.ones((GRID_H, GRID_W), dtype=float)

    # congestion: varje bil ökar cellens kostnad
    temp = np.zeros_like(cost)
    for car in cars:
        r, c = car.pos
        temp[r, c] += 1

    cost += temp * 0.8  # öka kostnad per bil på cellen
    return cost


def update():
    cost_map = compute_cost_map(cars)

    # STEG 1 – REGISTRERA ALLA UPPTAGNA POSITIONER
    old_occupied = {car.pos for car in cars}
    new_occupied = set()

    # -----------------------------
    # Ändring 1: lista för bilar som ska tas bort
    # -----------------------------
    cars_to_remove = []

    for car in cars:
        # Kolla om bilen har nått sitt mål
        if car.pos == car.goal:
            # -----------------------------
            # Ändring 2: markera bilen för borttagning
            # -----------------------------
            cars_to_remove.append(car)
            continue

        # STEG 2 – OMPROGRAMMERA RUTTER IBLAND
        car.recalc_delay -= 1
        if car.recalc_delay <= 0:
            car.recalc_path(cost_map)
            car.recalc_delay = random.randint(10, 25)

    # STEG 3 – FÖRSÖK FLYTTA ALLA BILAR
    for car in cars:
        # hoppa över bilar som ska tas bort
        if car in cars_to_remove:
            continue

        nxt = car.next_pos()

        # står still eller framme
        if nxt == car.pos:
            new_occupied.add(car.pos)
            continue

        # BLOCKERAD?
        if nxt in old_occupied or nxt in new_occupied:
            # bilen kan inte flytta, stannar
            new_occupied.add(car.pos)
            continue

        # FLYTTA
        car.pos = nxt
        car.index += 1
        new_occupied.add(car.pos)

    # -----------------------------
    # Ändring 3: ta bort bilar som nått mål
    # -----------------------------
    for car in cars_to_remove:
        if car in cars:
            cars.remove(car)

    # STEG 4 – UPPDATERA OCCUPIED
    old_occupied.clear()
    old_occupied.update(new_occupied)

# ----------------------------------
# 6. Visualization
# ----------------------------------

fig, ax = plt.subplots(figsize=(6,6))

def animate(_):
    # Kör update som tar bort bilar som nått mål
    update()

    # Rensa ax
    ax.clear()

    # Rita grid
    ax.imshow(grid, cmap="gray_r")

    # Rita bara bilar som finns kvar i cars
    if cars:  # säkerställ att listan inte är tom
        xs = [car.pos[1] for car in cars]
        ys = [car.pos[0] for car in cars]
        ax.scatter(xs, ys, c="red", s=30)

    # Title och ticks
    ax.set_title("Dynamic Traffic with Weighted Paths")
    ax.set_xticks([])
    ax.set_yticks([])

    return []

anim = FuncAnimation(fig, animate, frames=1000, interval=120)
plt.show()
