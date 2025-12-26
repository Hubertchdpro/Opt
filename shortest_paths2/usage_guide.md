# Usage Guide for Shortest Path Algorithms Implementation

This guide explains how to use the complete implementation of shortest path algorithms extracted from the comprehensive report. The code is consolidated in `complete_code.py` and includes implementations of Dijkstra, Bellman-Ford, Ford, and Floyd-Warshall algorithms.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Code](#running-the-code)
4. [Using Individual Components](#using-individual-components)
5. [Examples](#examples)
6. [Output Interpretation](#output-interpretation)
7. [Troubleshooting](#troubleshooting)

## Prerequisites {#prerequisites}

- **Python 3.6 or higher**
- **NumPy library** (required for matrix-based algorithms like Floyd-Warshall)
- Basic understanding of graph theory concepts

## Installation {#installation}

1. Ensure Python 3.6+ is installed on your system
2. Install the required dependency:

```bash
pip install numpy
```

3. Download or copy the `complete_code.py` file to your working directory

## Running the Code {#running-the-code}

### Quick Start

Execute the entire test suite to see all algorithms in action:

```bash
python complete_code.py
```

This will run the `tester_algorithmes()` function, which demonstrates:
- Dijkstra on a graph with positive weights
- Bellman-Ford on a graph with negative weights
- Ford algorithm (optimized Bellman-Ford)
- Floyd-Warshall for all-pairs shortest paths

### Expected Output

The program will display:
- Graph structures
- Algorithm results with distances and paths
- Performance comparisons
- Warning messages for negative cycles (if detected)

## Using Individual Components {#using-individual-components}

### 1. Creating a Graph

```python
from complete_code import Graphe

# Create a graph with 5 vertices (0 to 4)
g = Graphe(5)

# Add directed edges with weights
g.ajouter_arc(0, 1, 4.0)  # Edge from 0 to 1 with cost 4
g.ajouter_arc(0, 2, 2.0)  # Edge from 0 to 2 with cost 2
g.ajouter_arc(1, 3, 1.0)  # Edge from 1 to 3 with cost 1
```

### 2. Running Dijkstra Algorithm

**Requirements**: All edge weights must be non-negative.

```python
from complete_code import dijkstra, afficher_resultats

# Compute shortest paths from source vertex 0
distances, predecessors = dijkstra(g, 0)

# Display results
afficher_resultats(distances, predecessors, 0, g)
```

### 3. Running Bellman-Ford Algorithm

**Use when**: Graph may contain negative edge weights.

```python
from complete_code import bellman_ford

distances, predecessors, has_negative_cycle = bellman_ford(g, 0)

if has_negative_cycle:
    print("Warning: Negative cycle detected!")
else:
    afficher_resultats(distances, predecessors, 0, g)
```

### 4. Running Ford Algorithm

**Use when**: Optimized version of Bellman-Ford for graphs with negative weights.

```python
from complete_code import ford

distances, predecessors, has_negative_cycle = ford(g, 0)

if has_negative_cycle:
    print("Warning: Negative cycle detected!")
else:
    afficher_resultats(distances, predecessors, 0, g)
```

### 5. Running Floyd-Warshall Algorithm

**Use when**: Need shortest paths between all pairs of vertices.

```python
import numpy as np
from complete_code import floyd_warshall, reconstruire_chemin_floyd

# Create cost matrix (use float('inf') for no edge)
INF = float('inf')
cost_matrix = np.array([
    [0, 4, 2, INF],
    [INF, 0, 1, 3],
    [INF, INF, 0, 5],
    [INF, INF, INF, 0]
])

distances, next_vertices = floyd_warshall(cost_matrix)

# Get path from vertex 0 to vertex 3
path = reconstruire_chemin_floyd(next_vertices, 0, 3)
print(f"Shortest path: {' -> '.join(map(str, path))}")
print(f"Distance: {distances[0, 3]}")
```

### 6. Reconstructing Paths

For single-source algorithms (Dijkstra, Bellman-Ford, Ford):

```python
from complete_code import reconstruire_chemin

# Get path from source to target
path = reconstruire_chemin(predecessors, source, target)
if path:
    print(f"Path: {' -> '.join(map(str, path))}")
else:
    print("No path exists")
```

## Examples {#examples}

### Example 1: Basic Dijkstra Usage

```python
from complete_code import Graphe, dijkstra, afficher_resultats

# Create a simple graph
g = Graphe(4)
g.ajouter_arc(0, 1, 1)
g.ajouter_arc(0, 2, 4)
g.ajouter_arc(1, 2, 2)
g.ajouter_arc(1, 3, 5)
g.ajouter_arc(2, 3, 1)

# Run Dijkstra from vertex 0
distances, predecessors = dijkstra(g, 0)
afficher_resultats(distances, predecessors, 0, g)
```

### Example 2: Handling Negative Weights

```python
from complete_code import Graphe, bellman_ford, afficher_resultats

# Create graph with negative weights
g = Graphe(4)
g.ajouter_arc(0, 1, 4)
g.ajouter_arc(0, 2, 2)
g.ajouter_arc(1, 2, -1)  # Negative weight
g.ajouter_arc(2, 3, 1)

# Use Bellman-Ford
distances, predecessors, has_cycle = bellman_ford(g, 0)

if has_cycle:
    print("Graph contains a negative cycle")
else:
    afficher_resultats(distances, predecessors, 0, g)
```

### Example 3: All-Pairs Shortest Paths

```python
import numpy as np
from complete_code import floyd_warshall

# Define cost matrix
costs = np.array([
    [0, 3, 8, float('inf')],
    [float('inf'), 0, float('inf'), 1],
    [float('inf'), 4, 0, float('inf')],
    [2, float('inf'), -5, 0]
])

distances, next_vertices = floyd_warshall(costs)

print("All-pairs distances:")
print(distances)
```

## Output Interpretation {#output-interpretation}

### Distance Values
- **Finite number**: Shortest path distance from source
- **`inf`**: Vertex is unreachable from source
- **Negative number**: Possible with negative weights (Bellman-Ford/Ford)

### Path Reconstruction
- **Valid path**: List of vertices from source to destination
- **Empty list**: No path exists between vertices

### Algorithm-Specific Messages
- **"Circuit absorbant détecté"**: Negative cycle found (Bellman-Ford/Ford)
- **Graph structure display**: Shows all edges and their weights

### Performance Notes
- **Dijkstra**: Fastest for positive weights, O((n+m) log n) with heap
- **Bellman-Ford**: Handles negatives, O(n×m), detects negative cycles
- **Ford**: Optimized Bellman-Ford, better average performance
- **Floyd-Warshall**: All-pairs, O(n³), good for dense graphs

## Troubleshooting {#troubleshooting}

### Common Issues

1. **Import Error**: Ensure `complete_code.py` is in your Python path
2. **NumPy Not Found**: Install with `pip install numpy`
3. **Infinite Loop**: Check for negative cycles in Bellman-Ford/Ford
4. **Wrong Results**: Verify graph construction and edge directions

### Algorithm Selection Guide

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Positive weights only | Dijkstra |
| May have negative weights | Bellman-Ford or Ford |
| Need all pairs shortest paths | Floyd-Warshall |
| Large sparse graph | Dijkstra |
| Dense graph | Floyd-Warshall |
| Must detect negative cycles | Bellman-Ford or Ford |

### Performance Tips

- Use Dijkstra when possible (fastest for positive weights)
- For negative weights, Ford often performs better than Bellman-Ford
- Floyd-Warshall is ideal when you need all pairs (not just single source)
- Consider graph size: Floyd-Warshall may be slow for n > 1000

## Conclusion

This implementation provides a complete toolkit for shortest path problems. Start with the test function to understand the algorithms, then adapt the individual functions for your specific use cases. Remember to choose the appropriate algorithm based on your graph's characteristics (weights, density, requirements).
