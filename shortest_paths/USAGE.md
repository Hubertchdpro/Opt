# shortest_paths usage

Installation

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate    # on Unix/macOS
.venv\Scripts\activate      # on Windows (PowerShell)
```

2. Install dependencies:

```bash
pip install -r shortest_paths/requirements.txt
```

Running the examples

- Run the bundled tests/demo:

```bash
python -m shortest_paths.examples
```

Using the library

From Python code you can import and use the data structures and algorithms:

```python
from shortest_paths import Graphe, dijkstra, bellman_ford, ford, floyd_warshall

g = Graphe(4)
g.ajouter_arc(0,1,4)
# ... add arcs
distances, preds = dijkstra(g, 0)
```

Files

- `shortest_paths/graph.py` — Graphe class
- `shortest_paths/algorithms.py` — implementations of algorithms
- `shortest_paths/examples.py` — small demo and tests
