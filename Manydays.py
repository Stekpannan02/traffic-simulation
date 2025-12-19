import numpy as np
import random
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from heapq import heappush, heappop

# ------------------------------
# 0. Settings / global
# ------------------------------
GRID_W = 40
GRID_H = 40
roadworks = {}  # key=(r,c), value=timer kvar
ROADWORK_PROB = 0.00005
ROADWORK_DURATION = 50  # antal steg vägarbetet finns kvar
roadwork_events = []  # timesteps när ett nytt vägarbete startar

# ------------------------------
# 1. Road mask + two-lane structure
# ------------------------------
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
for i in range(9,32):
    road_mask[i,0] = 0
for j in range(0,8):
    road_mask[16,j] = 0
    road_mask[24,j] = 0
for i in range(33,39):
    road_mask[i,8] = 0

# Lanes
lanes = np.empty((GRID_H, GRID_W), dtype=object)
for r in range(GRID_H):
    for c in range(GRID_W):
        lanes[r, c] = {'N': [], 'S': [], 'E': [], 'W': [], 'still': []}

# ------------------------------
# Edge + fixed goals
# ------------------------------
SPAWN_POINTS_TEST = [(0, c) for c in range(GRID_W) if road_mask[0, c] == 1]
DEST_POINTS_TEST = [(GRID_W-1, 0), (GRID_W-1, 16), (GRID_H-1, GRID_W-1)]

# ------------------------------
# Dijkstra
# ------------------------------
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
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < H and 0 <= nc < W): 
                continue
            if road_mask[nr,nc] != 1 or (nr,nc) in roadworks:
                continue
            nxt = (nr,nc)
            new_cost = cur_cost + cost_map[nr,nc]
            if nxt not in dist or new_cost < dist[nxt]:
                dist[nxt] = new_cost
                visited[nxt] = cur
                heappush(pq, (new_cost,nxt))

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = visited.get(node)
    path.reverse()
    return path

# ------------------------------
# Car class
# ------------------------------
class Car:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.pos = start
        self.path = []
        self.index = 0
        self.last_dir = None
        self.stuck_counter = 0
        self.recalc_path(np.ones((GRID_H, GRID_W)))

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

# ------------------------------
# Intersections
# ------------------------------
intersections = []
for r in range(1, GRID_H-1):
    for c in range(1, GRID_W-1):
        if road_mask[r,c] == 1 and road_mask[r-1,c] == 1 and road_mask[r+1,c] == 1 \
           and road_mask[r,c-1] == 1 and road_mask[r,c+1] == 1:
            intersections.append((r,c))

# ------------------------------
# Traffic Light Class
# ------------------------------
class TrafficLight:
    def __init__(self, position, ns_green=40, ew_green=40):
        self.position = position
        self.ns_green = ns_green
        self.ew_green = ew_green
        self.timer = 0
        self.state = "NS_GREEN"

    def tick(self):
        self.timer += 1
        if self.state == "NS_GREEN":
            if self.timer >= self.ns_green:
                self.state = "EW_GREEN"
                self.timer = 0
        else:
            if self.timer >= self.ew_green:
                self.state = "NS_GREEN"
                self.timer = 0

    def allows(self, from_pos, to_pos):
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        if dr != 0:
            return self.state == "NS_GREEN"
        if dc != 0:
            return self.state == "EW_GREEN"
        return True

traffic_lights = [
    TrafficLight((16,16), ns_green=30, ew_green=30),
    TrafficLight((16,8), ns_green=50, ew_green=20),
    TrafficLight((32,16), ns_green=45, ew_green=15),
    TrafficLight((16,39), ns_green=45, ew_green=15),
    TrafficLight((24,24), ns_green=30, ew_green=15),
    TrafficLight((24,16), ns_green=30, ew_green=15),
]
traffic_light_map = {tl.position: tl for tl in traffic_lights}

# ------------------------------
# Colors for directions (för eventuella plots)
# ------------------------------
def dir_color(d):
    if d == 'N': return 'lightblue'
    if d == 'S': return 'lightgreen'
    if d == 'E': return 'orange'
    if d == 'W': return 'magenta'
    return 'red'

# ------------------------------
# Simulering utan animation
# ------------------------------
DAYS = 20
T_DAY = 24*60  # minuter per dag
congestion_data = np.zeros((DAYS, T_DAY), dtype=int)

# Reproducerbarhet
np.random.seed(42)
random.seed(42)

for day in range(DAYS):
    cars = []
    lanes = np.empty((GRID_H, GRID_W), dtype=object)
    for r in range(GRID_H):
        for c in range(GRID_W):
            lanes[r, c] = {'N': [], 'S': [], 'E': [], 'W': [], 'still': []}
    roadworks = {}
    roadwork_events = []
    timestep = 0
    congestion_history = []

    while timestep < T_DAY:
        # --- Spawn cars ---
        def traffic_intensity(t, T_max):
            x = t / T_max
            morning  = np.exp(-((x - 0.33)**2) / 0.003)
            evening  = np.exp(-((x - 0.70)**2) / 0.003)
            base = 0.05
            return base + morning + evening

        prob = traffic_intensity(timestep, T_DAY)
        n_to_spawn = np.random.poisson(prob)
        for _ in range(n_to_spawn):
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

        # --- Update cars & roadworks ---
        def compute_cost_map(cars):
            cost = np.ones((GRID_H, GRID_W), dtype=float)
            for car in cars:
                r, c = car.pos
                cnt = sum(len(v) for v in lanes[r,c].values())
                cost[r,c] += cnt*2
            return cost

        # Slumpa vägarbeten
        for r in range(GRID_H):
            for c in range(GRID_W):
                if road_mask[r,c] == 1 and (r,c) not in roadworks:
                    if random.random() < ROADWORK_PROB:
                        roadworks[(r,c)] = ROADWORK_DURATION
                        roadwork_events.append(timestep)
        # Minska timer
        to_remove = []
        for pos in roadworks:
            roadworks[pos] -= 1
            if roadworks[pos] <= 0:
                to_remove.append(pos)
        for pos in to_remove:
            roadworks.pop(pos)

        cost_map = compute_cost_map(cars)
        old_positions = {car: car.pos for car in cars}
        cars_to_remove = []
        desired_of = {}
        desired_to = {}

        for car in cars:
            if car.pos == car.goal:
                cars_to_remove.append(car)
                continue
            if car.pos in intersections and car.stuck_counter >= 2:
                car.recalc_path(cost_map)
                car.stuck_counter = 0
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
                if road_mask[r_new,c_new] != 1 or (r_new,c_new) in roadworks:
                    continue
                if car.pos in traffic_light_map:
                    tl = traffic_light_map[car.pos]
                    if not tl.allows(car.pos, desired):
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

        for car in cars_to_remove:
            r,c = car.pos
            for key in lanes[r,c]:
                if car in lanes[r,c][key]:
                    lanes[r,c][key].remove(car)
            if car in cars:
                cars.remove(car)

        # Stuck counters + congestion
        stopped = 0
        for car in cars:
            if old_positions[car] == car.pos:
                stopped += 1
                car.stuck_counter += 1
            else:
                car.stuck_counter = 0
        congestion_history.append(stopped)

        # Tick trafikljus
        for tl in traffic_lights:
            tl.tick()

        timestep += 1

    congestion_data[day,:] = congestion_history

# ------------------------------
# Medelvärde och standardavvikelse
# ------------------------------
avg_congestion = congestion_data.mean(axis=0)
std_congestion = congestion_data.std(axis=0)

# ------------------------------
# Plot
# ------------------------------
start_time = dt.datetime(2024, 1, 1, 0, 0)
times = [start_time + dt.timedelta(minutes=i) for i in range(T_DAY)]

fig, ax = plt.subplots(figsize=(12,6))

lower_bound = np.maximum(avg_congestion - std_congestion, 0)
upper_bound = avg_congestion + std_congestion

# Fill för standardavvikelse
ax.fill_between(times, lower_bound, upper_bound, color='blue', alpha=0.3, label='Std dev')

# Medelvärde
ax.plot(times, avg_congestion, label='Average congestion', color='blue', linewidth=2)

# Titlar och labels
ax.set_title("Average Congestion over 20 Days", fontsize=28, fontweight='bold')
ax.set_ylabel("Still Cars", fontsize=24)
ax.set_xlabel("Time of Day", fontsize=24)

# Ticks
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
important_hours = [8, 17]
tick_times = [start_time + dt.timedelta(hours=h) for h in important_hours]
ax.set_xticks(tick_times)
ax.tick_params(axis='both', labelsize=20)

# Legend
ax.legend(fontsize=20, loc='upper right')

# Autoformat för datum
fig.autofmt_xdate()
plt.tight_layout()
plt.show()


