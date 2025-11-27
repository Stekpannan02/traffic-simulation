import matplotlib.pyplot as plt
import math
import random

# ---- DEFINE NODES ----
nodes = {
    "A": (0, 0),
    "B": (1, 0.5),
    "C": (2, 4),
    "D": (3, 1),
}

# ---- DEFINE EDGES ----
edges = [
    ("A", "B"),
    ("B", "C"),
    ("B", "D"),
    ("C", "D")
]

# ---- FUNCTION TO CALCULATE DISTANCE ----
def distance(n1, n2):
    x1, y1 = nodes[n1]
    x2, y2 = nodes[n2]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# ---- CREATE GRAPH WITH DISTANCES ----
graph = {}
for n1, n2 in edges:
    d = distance(n1, n2)
    graph.setdefault(n1, []).append((n2, d))
    graph.setdefault(n2, []).append((n1, d))  # tvåvägs

# ---- STOCHASTIC AGENT PATH FUNCTION ----
def stochastic_agent_path(start, goal, graph, shortest_prob=0.9):
    current = start
    path = [current]
    visited = set()
    
    while current != goal:
        visited.add(current)
        neighbors = [(n, d) for n, d in graph[current] if n not in visited]
        if not neighbors:  # Om vi fastnar
            break
        
        # Sortera grannar efter avstånd till målet
        neighbors.sort(key=lambda x: distance(x[0], goal))
        if random.random() < shortest_prob:
            # 90% chans: välj närmaste
            next_node = neighbors[0][0]
        else:
            # 10% chans: välj slumpmässigt bland övriga
            if len(neighbors) > 1:
                next_node = random.choice(neighbors[1:])[0]
            else:
                next_node = neighbors[0][0]
        
        path.append(next_node)
        current = next_node
        
    return path

# Generera agentens väg
agent_path = stochastic_agent_path("A", "D", graph)
print("Agentens väg:", agent_path)

# ---- PLOT ----
plt.figure(figsize=(6, 6))

# Plot nodes
for name, (x, y) in nodes.items():
    plt.scatter(x, y)
    plt.text(x + 0.05, y + 0.05, name)

# Plot edges
for n1, n2 in edges:
    x1, y1 = nodes[n1]
    x2, y2 = nodes[n2]
    plt.plot([x1, x2], [y1, y2], color="gray")

# Plot agent path
for i in range(len(agent_path) - 1):
    x1, y1 = nodes[agent_path[i]]
    x2, y2 = nodes[agent_path[i + 1]]
    plt.plot([x1, x2], [y1, y2], color="red", linewidth=2, label="Agent path" if i==0 else "")

plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Node Graph with Stochastic Agent Path")
plt.legend()
plt.show()
