import math
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy.random as npr

import matplotlib.pyplot as plt

# ---- DEFINE NODES ----
# Format: "namn": (x, y)
nodes = {
    "A": (0, 0),
    "B": (1, 0.5),
    "C": (2, 4),
    "D": (3, 1),
}

# ---- DEFINE EDGES ----
# Format: ("nod1", "nod2")
edges = [
    ("A", "B"),
    ("B", "C"),
    ("A", "D"),
    ["A", "C"]
]

# ---- PLOT ----
plt.figure(figsize=(6, 6))

# Plot nodes
for name, (x, y) in nodes.items():
    plt.scatter(x, y)
    plt.text(x + 0.05, y + 0.05, name)  # label the node

# Plot edges
for n1, n2 in edges:
    x1, y1 = nodes[n1]
    x2, y2 = nodes[n2]
    plt.plot([x1, x2], [y1, y2])

plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("General Node Graph")
plt.show()
