# Utilisation de shortest_paths

Installation

1. Créez un environnement virtuel (recommandé) :

```bash
python -m venv .venv
# activer sous macOS / Linux
source .venv/bin/activate
# activer sous Windows (PowerShell)
.venv\Scripts\Activate.ps1
# activer sous Windows (CMD)
.venv\Scripts\activate.bat
```

2. Installez les dépendances :

```bash
pip install -r shortest_paths/requirements.txt
```

Exécuter les exemples

- Lancer la démo / les tests intégrés :

```bash
python -m shortest_paths.examples
```

Utilisation depuis Python

Importez les structures et fonctions puis utilisez-les :

```python
from shortest_paths import Graphe, dijkstra, bellman_ford, ford, floyd_warshall

g = Graphe(4)
g.ajouter_arc(0, 1, 4)
# ... ajouter des arcs
distances, preds = dijkstra(g, 0)
```

Fichiers principaux

- `shortest_paths/graph.py` — classe `Graphe`
- `shortest_paths/algorithms.py` — implémentations des algorithmes
- `shortest_paths/examples.py` — démonstration et petits tests
