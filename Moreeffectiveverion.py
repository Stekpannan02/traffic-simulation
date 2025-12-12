import numpy as np
import random
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from matplotlib.animation import FuncAnimation

# -------------------------
# 0. Settings / global
# -------------------------
GRID_W = 40
GRID_H = 40
SIM_STEPS = 400
roadworks = {}  # key=(r,c), value=timer kvar
ROADWORK_PROB = 0.0001
ROADWORK_DURATION = 50  # steg vägarbetet finns kvar
congestion_history = []

# -------------------------
# 1. Road mask + lanes
# -------------------------
road_mask = np.zeros((GRID_H, GRID_W), dtype=int)
for i in range(0, GRID_H, 8):
    road_mask[i, :] = 1
for j in range(0, GRID_W, 8):
    road_mask[:, j] = 1
road_mask[0, :] = road_mask[GRID_H-1, :] = 1
road_mask[:, 0] = road_mask[:, GRID_W-1] = 1

# Ta bort vissa mellanvägar
for i in range(1,8): road_mask[i,8] = 0
for i in range(9,32): road_mask[i,32] = 0
for j in range(25,39): road_mask[16,j] = 0; road_mask[24,j] = 0
for j in range(9,16): road_mask[24,j] = 0

lanes = np.empty((GRID_H, GRID_W), dtype=object)
for r in range(GRID_H):
    for c in range(GRID_W):
        lanes[r,c] = {'N': [], 'S': [], 'E': [], 'W': [], 'still': []}

# -------------------------
# Spawn & destination
# -------------------------
SPAWN_POINTS_TEST = [(0, c) for c in range(GRID_W) if road_mask[0,c]==1]
DEST_POINTS_TEST = [(GRID_H-1,0), (0,GRID_W-1), (32,16)]

# -------------------------
# Dijkstra
# -------------------------
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
        r,c = cur
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc = r+dr,c+dc
            if not (0<=nr<H and 0<=nc<W): continue
            if road_mask[nr,nc]!=1 or (nr,nc) in roadworks: continue
            nxt = (nr,nc)
            new_cost = cur_cost + cost_map[nr,nc]
            if nxt not in dist or new_cost < dist[nxt]:
                dist[nxt] = new_cost
                visited[nxt] = cur
                heappush(pq,(new_cost,nxt))
    path=[]
    node=goal
    while node is not None:
        path.append(node)
        node=visited.get(node)
    path.reverse()
    return path

# -------------------------
# Car class
# -------------------------
class Car:
    def __init__(self,start,goal):
        self.start=start
        self.goal=goal
        self.pos=start
        self.path=[]
        self.index=0
        self.last_dir=None
        self.stuck_counter=0
        self.recalc_path(np.ones((GRID_H,GRID_W)))

    def recalc_path(self,cost_map):
        self.path=dijkstra_path(self.pos,self.goal,cost_map)
        self.index=0

    def next_pos(self):
        if self.index+1<len(self.path):
            nxt=self.path[self.index+1]
            if nxt==self.pos: return self.goal
            return nxt
        return self.goal

    def planned_direction(self):
        nxt=self.next_pos()
        if nxt==self.pos: return None
        dr,nc=nxt[0]-self.pos[0], nxt[1]-self.pos[1]
        if dr==1: return 'S'
        if dr==-1: return 'N'
        dc=nxt[1]-self.pos[1]
        if dc==1: return 'E'
        if dc==-1: return 'W'
        return None

# -------------------------
# Intersections
# -------------------------
intersections=[]
for r in range(1,GRID_H-1):
    for c in range(1,GRID_W-1):
        if road_mask[r,c]==1 and road_mask[r-1,c]==1 and road_mask[r+1,c]==1 and road_mask[r,c-1]==1 and road_mask[r,c+1]==1:
            intersections.append((r,c))

# -------------------------
# Traffic lights
# -------------------------
class TrafficLight:
    def __init__(self,position,ns_green=40,ew_green=40):
        self.position=position
        self.ns_green=ns_green
        self.ew_green=ew_green
        self.timer=0
        self.state="NS_GREEN"
    def tick(self):
        self.timer+=1
        if self.state=="NS_GREEN":
            if self.timer>=self.ns_green:
                self.state="EW_GREEN"; self.timer=0
        else:
            if self.timer>=self.ew_green:
                self.state="NS_GREEN"; self.timer=0
    def allows(self,from_pos,to_pos):
        dr,to_dc=to_pos[0]-from_pos[0],to_pos[1]-from_pos[1]
        if dr!=0: return self.state=="NS_GREEN"
        if to_dc!=0: return self.state=="EW_GREEN"
        return True

traffic_lights=[
    TrafficLight((16,16),30,30),
    TrafficLight((16,8),50,20)
]
traffic_light_map={tl.position:tl for tl in traffic_lights}

# -------------------------
# Cost map
# -------------------------
def compute_cost_map(cars):
    cost=np.ones((GRID_H,GRID_W),dtype=float)
    for car in cars:
        r,c=car.pos
        cnt=sum(len(v) for v in lanes[r,c].values())
        cost[r,c]+=cnt*2
    return cost

# -------------------------
# Spawning
# -------------------------
T_DAY=24*60
timestep=0
cars=[]
def traffic_intensity(t,T_max):
    x=t/T_max
    morning=np.exp(-((x-0.33)**2)/0.003)
    evening=np.exp(-((x-0.70)**2)/0.003)
    base=0.05
    return base+morning+evening

def spawn_cars(timestep):
    current_minute=timestep%T_DAY
    prob=traffic_intensity(current_minute,T_DAY)
    n_to_spawn=np.random.poisson(prob)
    for _ in range(n_to_spawn):
        start=random.choice(SPAWN_POINTS_TEST)
        goal=random.choice(DEST_POINTS_TEST)
        if goal==start: continue
        temp_car=Car(start,goal)
        if len(temp_car.path)<=1: continue
        car=temp_car
        cars.append(car)
        r,c=start
        lanes[r,c]['still'].append(car)

# -------------------------
# Update
# -------------------------
def update_with_lanes():
    global congestion_history
    cost_map=compute_cost_map(cars)
    old_positions={car:car.pos for car in cars}
    cars_to_remove=[]
    # Vägarbeten
    for r in range(GRID_H):
        for c in range(GRID_W):
            if road_mask[r,c]==1 and (r,c) not in roadworks:
                if random.random()<ROADWORK_PROB: roadworks[(r,c)]=ROADWORK_DURATION
    # Minska timer
    to_remove=[]
    for pos in roadworks:
        roadworks[pos]-=1
        if roadworks[pos]<=0: to_remove.append(pos)
    for pos in to_remove: roadworks.pop(pos)
    # Recalc
    for car in cars:
        if car.stuck_counter>=2: car.recalc_path(cost_map); car.stuck_counter=0

    desired_of={}
    desired_to={}
    for car in cars:
        if car.pos==car.goal: cars_to_remove.append(car); continue
        if car.pos in intersections: car.recalc_path(cost_map)
        nxt=car.next_pos()
        lane=car.planned_direction()
        desired_of[car]=(nxt,lane)
        desired_to.setdefault((nxt,lane),[]).append(car)

    moved=set()
    for target,contenders in desired_to.items():
        if len(contenders)>1: contenders=[random.choice(contenders)]
        for car in contenders:
            if car in moved or car in cars_to_remove: continue
            desired,lane=desired_of[car]
            r_new,c_new=desired
            if desired==car.pos: continue
            if not(0<=r_new<GRID_H and 0<=c_new<GRID_W): continue
            if road_mask[r_new,c_new]!=1 or (r_new,c_new) in roadworks: continue
            if car.pos in traffic_light_map:
                tl=traffic_light_map[car.pos]
                if not tl.allows(car.pos,desired): continue
            occ=lanes[r_new,c_new][lane] if lane else lanes[r_new,c_new]['still']
            if len(occ)==0:
                r_old,c_old=car.pos
                for key in lanes[r_old,c_old]:
                    if car in lanes[r_old,c_old][key]: lanes[r_old,c_old][key].remove(car)
                if lane: lanes[r_new,c_new][lane].append(car)
                else: lanes[r_new,c_new]['still'].append(car)
                car.pos=(r_new,c_new)
                car.index+=1
                car.last_dir=lane
                moved.add(car)
    for car in cars_to_remove:
        r,c=car.pos
        for key in lanes[r,c]:
            if car in lanes[r,c][key]: lanes[r,c][key].remove(car)
        if car in cars: cars.remove(car)

    stopped=0
    for car in cars:
        if old_positions[car]==car.pos: stopped+=1; car.stuck_counter+=1
        else: car.stuck_counter=0
    congestion_history.append(stopped)

# -------------------------
# Visualization
# -------------------------
fig,ax=plt.subplots(figsize=(8,8))
ax.imshow(road_mask,cmap="gray_r")
ax.set_xticks([]); ax.set_yticks([])

# Scatter för bilar och vägarbeten
car_positions=np.array([car.pos[::-1] for car in cars])
scat_cars=ax.scatter(car_positions[:,0],car_positions[:,1],c='red',s=20)
scat_roadworks=ax.scatter([],[],c='brown',s=30,marker='s')

# Trafikljus scatter
tl_positions=np.array([tl.position[::-1] for tl in traffic_lights])
tl_colors=np.array(['green' if tl.state=="NS_GREEN" else 'red' for tl in traffic_lights])
scat_tl=ax.scatter(tl_positions[:,0],tl_positions[:,1],c=tl_colors,s=80,marker='o')

# Animation

timestep=0
def animate(frame):
    global timestep
    spawn_cars(timestep)
    update_with_lanes()

    # Uppdatera car scatter
    car_positions=np.array([car.pos[::-1] for car in cars])
    scat_cars.set_offsets(car_positions)

    # Uppdatera roadworks scatter
    if roadworks:
        rw_positions=np.array([pos[::-1] for pos in roadworks.keys()])
        scat_roadworks.set_offsets(rw_positions)
    else:
        scat_roadworks.set_offsets(np.zeros((0,2)))

    # Uppdatera trafikljus färger
    tl_colors=np.array(['green' if tl.state=="NS_GREEN" else 'red' for tl in traffic_lights])
    scat_tl.set_color(tl_colors)
    for tl in traffic_lights: tl.tick()

    # Tid
    minutes=timestep%T_DAY
    hours=minutes//60
    mins=minutes%60
    ax.set_title(f"Time {hours:02d}:{mins:02d} | Cars: {len(cars)}")
    timestep+=1
    return scat_cars,scat_roadworks,scat_tl

anim=FuncAnimation(fig,animate,frames=SIM_STEPS,interval=80,blit=False)
plt.show()

# Plot congestion
plt.figure()
plt.plot(congestion_history)
plt.title("Congestion Index Over Time")
plt.xlabel("Timestep")
plt.ylabel("Still Cars")
plt.show()
