import numpy as np
import random
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from matplotlib.animation import FuncAnimation

# ----------------------------------
# 0. Settings / global
# ----------------------------------
GRID_W = 40
GRID_H = 40
SIM_STEPS = 400

congestion_history = []

# ----------------------------------
# 1. Road mask + two-lane structure
# ----------------------------------
road_mask = np.zeros((GRID_H, GRID_W), dtype=int)

# Grundrutnät: var 8:e rad och kolumn
for i in range(0, GRID_H, 8):
    road_mask[i, :] = 1
for j in range(0, GRID_W, 8):
    road_mask[:, j] = 1

# Behåll kantvägarna
road_mask[0, :] = 1
road_mask[GRID_H-1, :] = 1
road_mask[:, 0] = 1
road_mask[:, GRID_W-1] = 1

# Ta bort några mellanvägar slumpmässigt (ej kantvägar)

            
for i in range(1, 8):
    road_mask[i,8] = 0
for i in range(9,32):
    road_mask[i,32] = 0
for j in range(25,39):
    road_mask[16,j] = 0
    road_mask[24,j] = 0

for j in range(9,16):
    road_mask[24,j] = 0
    
#road_mask
lanes = np.empty((GRID_H, GRID_W), dtype=object)
for r in range(GRID_H):
    for c in range(GRID_W):
        lanes[r, c] = {'N': [], 'S': [], 'E': [], 'W': [], 'still': []}

# ----------------------------------
# Edge + fixed goals
# ----------------------------------
SPAWN_POINTS_TEST = [(0, c) for c in range(GRID_W) if road_mask[0, c] == 1]
DEST_POINTS_TEST = [(GRID_H-1, 0), (0, GRID_W-1), (16,16)]

# ----------------------------------
# Dijkstra
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
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc = r+dr, c+dc
            if not (0 <= nr < H and 0 <= nc < W): 
                continue
            if road_mask[nr,nc] != 1:
                continue
            
            nxt = (nr,nc)
            new_cost = cur_cost + cost_map[nr,nc]
            if nxt not in dist or new_cost < dist[nxt]:
                dist[nxt] = new_cost
                visited[nxt] = cur
                heappush(pq,(new_cost,nxt))

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = visited.get(node)
    path.reverse()
    return path

# ----------------------------------
# Car class
# ----------------------------------
class Car:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.pos = start
        self.path = []
        self.index = 0
        self.last_dir = None
        self.stuck_counter = 0

        self.recalc_path(np.ones((GRID_H,GRID_W)))

    def recalc_path(self, cost_map):
        self.path = dijkstra_path(self.pos, self.goal, cost_map)
        self.index = 0

    def next_pos(self):
        if self.index + 1 < len(self.path):
            nxt = self.path[self.index+1]
            if nxt == self.pos:
                return self.goal
            return nxt
        return self.goal

    def planned_direction(self):
        nxt = self.next_pos()
        if nxt == self.pos:
            return None
        dr = nxt[0] - self.pos[0]
        dc = nxt[1] - self.pos[1]
        if dr == 1: return 'S'
        if dr == -1: return 'N'
        if dc == 1: return 'E'
        if dc == -1: return 'W'
        return None

# ----------------------------------
# Intersections
# ----------------------------------
intersections = []
for r in range(1, GRID_H-1):
    for c in range(1, GRID_W-1):
        if road_mask[r,c] == 1 and road_mask[r-1,c] == 1 and road_mask[r+1,c] == 1 \
           and road_mask[r,c-1] == 1 and road_mask[r,c+1] == 1:
            intersections.append((r,c))

# ----------------------------------
# Cost map
# ----------------------------------
def compute_cost_map(cars):
    cost = np.ones((GRID_H, GRID_W), dtype=float)
    for car in cars:
        r,c = car.pos
        cnt = sum(len(v) for v in lanes[r,c].values())
        cost[r,c] += cnt
    return cost

# ----------------------------------
# Spawning
# ----------------------------------
cars = []
def spawn_cars_test(n=1):
    for _ in range(n):
        start = random.choice(SPAWN_POINTS_TEST)
        goal = random.choice(DEST_POINTS_TEST)
        if goal == start:
            continue
        temp_car = Car(start, goal)
        if len(temp_car.path) <= 1:
            continue
        car = temp_car
        cars.append(car)
        r,c = start
        lanes[r,c]['still'].append(car)

spawn_cars_test(10)

# ----------------------------------
# UPDATE with lanes + stuck detection
# ----------------------------------
def update_with_lanes():
    global congestion_history
    cost_map = compute_cost_map(cars)
    old_positions = {car: car.pos for car in cars}

    cars_to_remove = []

    # Recalc if stuck 2 steg
    for car in cars:
        if car.stuck_counter >= 2:
            car.recalc_path(cost_map)
            car.stuck_counter = 0

    desired_of = {}
    desired_to = {}

    for car in cars:
        if car.pos == car.goal:
            cars_to_remove.append(car)
            continue
        if car.pos in intersections:
            car.recalc_path(cost_map)
        nxt = car.next_pos()
        lane = car.planned_direction()
        desired_of[car] = (nxt, lane)
        desired_to.setdefault((nxt, lane), []).append(car)

    moved = set()

    for target, contenders in desired_to.items():
        if len(contenders) > 1:
            car_to_move = random.choice(contenders)
            contenders = [car_to_move]
        for car in contenders:
            if car in moved or car in cars_to_remove:
                continue
            desired, lane = desired_of[car]
            r_new, c_new = desired
            if desired == car.pos:
                continue
            if not (0 <= r_new < GRID_H and 0 <= c_new < GRID_W): 
                continue
            if road_mask[r_new,c_new] != 1:
                continue
            occ = lanes[r_new,c_new][lane] if lane else lanes[r_new,c_new]['still']
            if len(occ) == 0:
                r_old,c_old = car.pos
                for key in lanes[r_old,c_old]:
                    if car in lanes[r_old,c_old][key]:
                        lanes[r_old,c_old][key].remove(car)
                if lane:
                    lanes[r_new,c_new][lane].append(car)
                else:
                    lanes[r_new,c_new]['still'].append(car)
                car.pos = (r_new,c_new)
                car.index += 1
                car.last_dir = lane
                moved.add(car)

    # Remove finished cars
    for car in cars_to_remove:
        r,c = car.pos
        for key in lanes[r,c]:
            if car in lanes[r,c][key]:
                lanes[r,c][key].remove(car)
        if car in cars:
            cars.remove(car)

    # Update stuck counters + congestion
    stopped = 0
    for car in cars:
        if old_positions[car] == car.pos:
            stopped += 1
            car.stuck_counter += 1
        else:
            car.stuck_counter = 0
    congestion_history.append(stopped)

# ----------------------------------
# Colors for directions
# ----------------------------------
def dir_color(d):
    if d == 'N': return 'blue'
    if d == 'S': return 'green'
    if d == 'E': return 'orange'
    if d == 'W': return 'purple'
    return 'red'

# ----------------------------------
# Visualization
# ----------------------------------
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(road_mask, cmap="gray_r")
ax.set_xticks([]); ax.set_yticks([])

dots = []
for car in cars:
    dot, = ax.plot([car.pos[1]],[car.pos[0]],'o',color=dir_color(car.last_dir),markersize=5)
    dots.append(dot)

goal_dots = []
for r,c in DEST_POINTS_TEST:
    gd, = ax.plot(c,r,'o',color='blue',markersize=8)
    goal_dots.append(gd)

def animate(step):
    spawn_cars_test(random.randint(0,2))
    update_with_lanes()

    while len(dots) > len(cars):
        d = dots.pop()
        d.remove()
    while len(dots) < len(cars):
        car = cars[len(dots)]
        d, = ax.plot([car.pos[1]],[car.pos[0]],'o',color=dir_color(car.last_dir),markersize=5)
        dots.append(d)
    for d,car in zip(dots,cars):
        d.set_data([car.pos[1]],[car.pos[0]])
        d.set_color(dir_color(car.last_dir))

    ax.set_title(f"Step {step} | Cars: {len(cars)}")
    return dots + goal_dots

anim = FuncAnimation(fig, animate, frames=SIM_STEPS, interval=80, blit=False)
plt.show()

# Plot congestion
plt.figure()
plt.plot(congestion_history)
plt.title("Congestion Index Over Time")
plt.xlabel("Timestep")
plt.ylabel("Still Cars")
plt.show()
