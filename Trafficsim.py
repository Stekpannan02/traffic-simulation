
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
congestion_history = []
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

# --------------------------
# Spawn & destination points
# --------------------------

def edge_cells():
    pts = []
    H, W = GRID_H, GRID_W
    for r in range(H):
        for c in range(W):
            if grid[r,c] == 1 and (r == 0 or r == H-1 or c == 0 or c == W-1):
                pts.append((r,c))
    return pts

SPAWN_POINTS = edge_cells()
DEST_POINTS  = SPAWN_POINTS[:]  # samma lista för enkelhet
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


# --------------------------
# Trafikintensitet under dagen
# --------------------------

T_DAY = 1000   # total simulation runtime
MAX_SPAWN_RATE = 2.5   # justerbar (ungefär max 2–4 bilar per timestep)

def traffic_intensity(t, T_max):
    x = t / T_max

    morning  = np.exp(-((x - 0.33)**2) / 0.002)
    evening  = np.exp(-((x - 0.70)**2) / 0.003)

    base = 0.05
    return base + morning + evening


def spawn_cars(t):
    expected = traffic_intensity(t, T_DAY) * MAX_SPAWN_RATE
    n_new = np.random.poisson(expected)

    for _ in range(n_new):
        start = random.choice(SPAWN_POINTS)
        dest  = random.choice(DEST_POINTS)

        if dest == start:
            continue

        cars.append(Car(start, dest))

def update():
    cost_map = compute_cost_map(cars)

    ### NY ###
    old_positions = [car.pos for car in cars]

    # STEG 1 – REGISTRERA ALLA UPPTAGNA POSITIONER
    old_occupied = {car.pos for car in cars}
    new_occupied = set()

    cars_to_remove = []
    
    for car in cars:
        if car.pos == car.goal:
            if car.pos in old_occupied:
                old_occupied.remove(car.pos)
            cars_to_remove.append(car)
            continue

        car.recalc_delay -= 1
        if car.recalc_delay <= 0:
            car.recalc_path(cost_map)
            car.recalc_delay = random.randint(10, 25)

    for car in cars:
        nxt = car.next_pos()

        if nxt == car.pos:
            new_occupied.add(car.pos)
            continue

        if nxt in old_occupied or nxt in new_occupied:
            new_occupied.add(car.pos)
            continue

        car.pos = nxt
        car.index += 1
        new_occupied.add(car.pos)

    old_occupied.clear()
    old_occupied.update(new_occupied)

    ### NY ###
    stopped = sum(
    1 for car, old_pos in zip(cars, old_positions)
    if car.pos == old_pos
)
    
    for car in cars_to_remove:
        cars.remove(car)
    congestion_history.append(stopped)

# ----------------------------------
# 6. Visualization
# ----------------------------------

fig, ax = plt.subplots(figsize=(6,6))

def animate(_):
    # Kör update som tar bort bilar som nått mål
    update()
    spawn_cars(_)   # _ = timestep

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


plt.figure()
plt.plot(congestion_history)
plt.title("Congestion Index Over Time")
plt.xlabel("Timestep")
plt.ylabel("Cars Standing Still")
plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from heapq import heappush, heappop
import random

# ----------------------------------
# 0. Settings / global
# ----------------------------------
GRID_W = 40
GRID_H = 40
SIM_STEPS = 1000
MAX_SPAWN_RATE = 3.0

T_DAY = 24 * 60  # simulera hela dagen i minuter
congestion_history = []
timestep = 0

# ----------------------------------
# 1. Road mask + grid (tvåfiligt, färre korsningar)
# ----------------------------------
road_mask = np.zeros((GRID_H, GRID_W), dtype=int)

for i in range(0, GRID_H, 4):
    if i % 8 == 0:  # varannan rad vägen borta för färre korsningar
        road_mask[i, :] = 1
for j in range(0, GRID_W, 4):
    if j % 8 == 0:
        road_mask[:, j] = 1

# grid: varje cell innehåller lista av bilar
grid = np.empty((GRID_H, GRID_W), dtype=object)
for r in range(GRID_H):
    for c in range(GRID_W):
        grid[r, c] = []

# ----------------------------------
# 1b. Kantceller (spawn + dest)
# ----------------------------------
def edge_cells():
    pts = []
    H, W = GRID_H, GRID_W
    for r in range(H):
        for c in range(W):
            if road_mask[r, c] == 1 and (r == 0 or r == H-1 or c == 0 or c == W-1):
                pts.append((r, c))
    return pts

SPAWN_POINTS = edge_cells()
DEST_POINTS = SPAWN_POINTS[:]

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
            if 0 <= nr < H and 0 <= nc < W and road_mask[nr, nc] == 1:
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
# 3. Car class med path
# ----------------------------------
class Car:
    def __init__(self, start, end):
        self.start = start
        self.goal = end
        self.pos = start
        self.path = []
        self.index = 0
        self.recalc_delay = random.randint(10,25)

    def recalc_path(self, cost_map):
        self.path = dijkstra_path(self.pos, self.goal, cost_map)
        self.index = 0

    def next_pos(self):
        # Returnera nästa cell längs path
        if self.index + 1 < len(self.path):
            return self.path[self.index + 1]
        return self.pos

# ----------------------------------
# 4. Spawning med dagskurva
# ----------------------------------
def traffic_intensity(t, T_max):
    x = t / T_max
    morning  = np.exp(-((x - 0.33)**2) / 0.002)
    evening  = np.exp(-((x - 0.70)**2) / 0.003)
    base = 0.05
    return base + morning + evening

def spawn_cars(t):
    expected = traffic_intensity(t, SIM_STEPS) * MAX_SPAWN_RATE
    n_new = np.random.poisson(expected)
    spawned = 0
    for _ in range(n_new):
        start = random.choice(SPAWN_POINTS)
        dest = random.choice(DEST_POINTS)
        if dest == start:
            continue
        car = Car(start, dest)
        cars.append(car)
        grid[start].append(car)
        spawned += 1
    return spawned

# ----------------------------------
# 5. Initiera bilar lite
# ----------------------------------
N_INIT = 20
cars = []
for _ in range(N_INIT):
    s = random.choice(SPAWN_POINTS)
    d = random.choice(DEST_POINTS)
    if d == s: continue
    c = Car(s,d)
    cars.append(c)
    grid[s].append(c)

# ----------------------------------
# 6. Kostkarta (congestion)
# ----------------------------------
def compute_cost_map(cars):
    cost = np.ones((GRID_H, GRID_W), dtype=float)
    for car in cars:
        r,c = car.pos
        cost[r,c] += 1
    return cost

# ----------------------------------
# 7. UPDATE med högertrafik
# ----------------------------------
def update():
    global congestion_history
    cost_map = compute_cost_map(cars)
    old_positions = [car.pos for car in cars]
    cars_to_remove = []

    # kontrollera framme
    for car in cars:
        if car.pos == car.goal:
            r,c = car.pos
            if car in grid[r,c]:
                grid[r,c].remove(car)
            cars_to_remove.append(car)

    # omplanering
    for car in cars:
        if car in cars_to_remove: continue
        car.recalc_delay -= 1
        if car.recalc_delay <=0:
            car.recalc_path(cost_map)
            car.recalc_delay = random.randint(10,25)

    # flytta bilar (högertrafik på raka segment, korsning = båda körfält)
    for car in cars:
        if car in cars_to_remove: continue
        nxt = car.next_pos()
        if nxt == car.pos: continue
        nr,nc = nxt
        # säkerhet: bara på vägar
        if road_mask[nr,nc]==0: continue
        # flytta bilen
        grid[car.pos[0], car.pos[1]].remove(car)
        grid[nr,nc].append(car)
        car.pos = nxt
        car.index += 1
        # check om mål nått
        if car.pos == car.goal:
            cars_to_remove.append(car)
            grid[nr,nc].remove(car)

    # ta bort bilar
    for car in cars_to_remove:
        if car in cars: cars.remove(car)

    # congestion
    stopped = sum(1 for car, old in zip(cars, old_positions) if car.pos == old)
    congestion_history.append(stopped)

# ----------------------------------
# 8. Visualization
# ----------------------------------
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(road_mask, cmap="gray_r")
ax.set_xticks([]); ax.set_yticks([])
dots = []
for car in cars:
    dot, = ax.plot([car.pos[1]], [car.pos[0]], 'o', color='red', markersize=5)
    dots.append(dot)

def animate(step):
    global timestep, dots
    spawn_cars(timestep)
    update()
    # uppdatera dots
    while len(dots) > len(cars):
        dot = dots.pop()
        dot.remove()
    while len(dots) < len(cars):
        car = cars[len(dots)]
        dot, = ax.plot([car.pos[1]], [car.pos[0]], 'o', color='red', markersize=5)
        dots.append(dot)
    for dot, car in zip(dots, cars):
        dot.set_data([car.pos[1]], [car.pos[0]])
    # klocka
    minutes = int((timestep/SIM_STEPS)*(T_DAY-1))
    hours = minutes//60
    mins = minutes%60
    ax.set_title(f"Time {hours:02d}:{mins:02d} | Cars: {len(cars)}")
    timestep +=1
    return dots

anim = FuncAnimation(fig, animate, frames=SIM_STEPS, interval=80, blit=False)
plt.show()

plt.figure()
plt.plot(congestion_history)
plt.title("Congestion Index Over Time")
plt.xlabel("Timestep")
plt.ylabel("Cars Standing Still")
plt.show()
