

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

