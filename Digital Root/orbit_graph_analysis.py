import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

def digit_sum(n):
    return sum(int(d) for d in str(n))

def digital_root(n):
    while n >= 10:
        n = digit_sum(n)
    return n

def f(n):
    return n + digit_sum(n) if n % 2 else n - digit_sum(n)

def build_orbit_graph(N):
    edges = []
    dr_sequences = {}
    for n in range(1, N+1):
        orbit = []
        dr_orbit = []
        visited = set()
        current = n
        while current not in visited and current != 0:
            visited.add(current)
            orbit.append(current)
            dr_orbit.append(digital_root(current))
            next_n = f(current)
            edges.append((current, next_n))
            current = next_n
        dr_sequences[n] = dr_orbit
    return edges, dr_sequences

if __name__ == "__main__":
    N = 1000  # You can increase for deeper analysis
    edges, dr_sequences = build_orbit_graph(N)
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Compute strongly connected components (SCCs)
    sccs = list(nx.strongly_connected_components(G))
    scc_sizes = [len(scc) for scc in sccs]
    scc_dr = [set(digital_root(n) for n in scc) for scc in sccs]

    # Plot SCC size distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(scc_sizes, bins=30)
    plt.xlabel('SCC Size')
    plt.ylabel('Frequency')
    plt.title('Strongly Connected Component Size Distribution')
    plt.tight_layout()
    plt.show()

    # Plot digital root distribution in SCCs
    dr_counts = pd.Series([dr for scc in sccs for dr in set(digital_root(n) for n in scc)])
    plt.figure(figsize=(8, 5))
    sns.countplot(x=dr_counts)
    plt.xlabel('Digital Root in SCC')
    plt.ylabel('Count')
    plt.title('Digital Root Distribution in SCCs')
    plt.tight_layout()
    plt.show()

    # Visualize a subgraph colored by digital root
    sample_nodes = list(range(1, min(200, N+1)))
    subG = G.subgraph(sample_nodes)
    node_colors = [digital_root(n) for n in subG.nodes()]
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(subG, seed=42)
    nodes = nx.draw_networkx_nodes(subG, pos, node_color=node_colors, cmap=plt.cm.plasma, node_size=40, ax=ax)
    nx.draw_networkx_edges(subG, pos, ax=ax, alpha=0.5)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=1, vmax=9))
    cbar = fig.colorbar(sm, ax=ax, label='Digital Root')
    plt.title('Orbit Graph Colored by Digital Root (Sample)')
    plt.tight_layout()
    plt.show()

    # Analyze and plot distance to 9 in orbits
    dr_distances = []
    for dr_orbit in dr_sequences.values():
        dr_distances.extend([9 - dr for dr in dr_orbit])
    plt.figure(figsize=(8, 5))
    sns.histplot(dr_distances, bins=9, discrete=True)
    plt.xlabel('Distance to 9 (9 - digital root)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distance to 9 in Orbits')
    plt.tight_layout()
    plt.show()

# --- Advanced Analyses ---

# 1. Cycle Structure Analysis
print("\n--- Cycle Structure Analysis ---")
cycles = set()
cycle_lengths = []
cycle_dr_sets = []
cycle_members = {}
visited_global = set()
for n in range(1, N+1):
    if n in visited_global:
        continue
    orbit = []
    visited = {}
    current = n
    while current not in visited and current != 0:
        visited[current] = len(orbit)
        orbit.append(current)
        current = f(current)
    if current != 0:
        cycle_start = visited[current]
        cycle = tuple(orbit[cycle_start:])
        cycles.add(cycle)
        cycle_lengths.append(len(cycle))
        dr_set = set(digital_root(x) for x in cycle)
        cycle_dr_sets.append(dr_set)
        for x in cycle:
            cycle_members.setdefault(cycle, set()).add(x)
        visited_global.update(cycle)
print(f"Number of unique cycles: {len(cycles)}")
print(f"Cycle lengths (sample): {cycle_lengths[:10]}")
print(f"Digital roots in cycles (sample): {cycle_dr_sets[:10]}")

# 2. Attractor Basin Size
print("\n--- Attractor Basin Size ---")
basin_counts = {cycle: 0 for cycle in cycles}
for n in range(1, N+1):
    orbit = []
    visited = set()
    current = n
    while current not in visited and current != 0:
        visited.add(current)
        orbit.append(current)
        current = f(current)
    if current != 0:
        cycle_start = orbit.index(current)
        cycle = tuple(orbit[cycle_start:])
        if cycle in basin_counts:
            basin_counts[cycle] += 1
basin_sizes = list(basin_counts.values())
plt.figure(figsize=(8, 5))
sns.histplot(basin_sizes, bins=30)
plt.xlabel('Basin Size')
plt.ylabel('Number of Cycles')
plt.title('Distribution of Attractor Basin Sizes')
plt.tight_layout()
plt.show()

# 3. Digital Root Transition Matrix
print("\n--- Digital Root Transition Matrix ---")
dr_matrix = pd.DataFrame(0, index=range(1,10), columns=range(1,10))
for n in range(1, N+1):
    dr_n = digital_root(n)
    dr_f = digital_root(f(n))
    if dr_n == 0 or dr_f == 0:
        continue
    dr_matrix.loc[dr_n, dr_f] += 1
plt.figure(figsize=(8, 6))
sns.heatmap(dr_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('digital_root(f(n))')
plt.ylabel('digital_root(n)')
plt.title('Digital Root Transition Matrix')
plt.tight_layout()
plt.show()

# 4. Longest Path to Digital Root 9
print("\n--- Longest Path to Digital Root 9 ---")
steps_to_9 = []
for n in range(1, N+1):
    current = n
    steps = 0
    seen = set()
    while digital_root(current) != 9 and current not in seen and current != 0:
        seen.add(current)
        current = f(current)
        steps += 1
    if digital_root(current) == 9:
        steps_to_9.append(steps)
plt.figure(figsize=(8, 5))
sns.histplot(steps_to_9, bins=30)
plt.xlabel('Steps to reach digital root 9')
plt.ylabel('Frequency')
plt.title('Distribution of Steps to Digital Root 9')
plt.tight_layout()
plt.show()

# Highlight numbers that take the longest to reach digital root 9
if steps_to_9:
    max_steps = max(steps_to_9)
    hardest = [n for n in range(1, N+1)
               if any(digital_root(x) != 9 for x in dr_sequences[n][:-1]) and len(dr_sequences[n])-1 == max_steps]
    print(f"Numbers that take the longest ({max_steps} steps) to reach digital root 9: {hardest}")
