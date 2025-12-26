# Guide d'Utilisation de l'Implémentation des Algorithmes de Chemins les Plus Courts

Ce guide explique comment utiliser l'implémentation complète des algorithmes de chemins les plus courts extraite du rapport complet. Le code est consolidé dans `complete_code.py` et inclut les implémentations de Dijkstra, Bellman-Ford, Ford et Floyd-Warshall.

## Table des Matières

1. [Prérequis](#prerequisites)
2. [Installation](#installation)
3. [Exécution du Code](#running-the-code)
4. [Utilisation des Composants Individuels](#using-individual-components)
5. [Exemples](#examples)
6. [Interprétation des Sorties](#output-interpretation)
7. [Dépannage](#troubleshooting)

## Prérequis {#prerequisites}

- **Python 3.6 ou supérieur**
- **Bibliothèque NumPy** (requise pour les algorithmes matriciels comme Floyd-Warshall)
- Compréhension de base des concepts de théorie des graphes

## Installation {#installation}

1. Assurez-vous que Python 3.6+ est installé sur votre système
2. Installez la dépendance requise :

```bash
pip install numpy
```

3. Téléchargez ou copiez le fichier `complete_code.py` dans votre répertoire de travail

## Exécution du Code {#running-the-code}

### Démarrage Rapide

Exécutez la suite de tests complète pour voir tous les algorithmes en action :

```bash
python complete_code.py
```

Cela exécutera la fonction `tester_algorithmes()`, qui démontre :
- Dijkstra sur un graphe avec poids positifs
- Bellman-Ford sur un graphe avec poids négatifs
- Algorithme de Ford (Bellman-Ford optimisé)
- Floyd-Warshall pour les chemins les plus courts toutes paires

### Sortie Attendue

Le programme affichera :
- Structures de graphes
- Résultats des algorithmes avec distances et chemins
- Comparaisons de performances
- Messages d'avertissement pour les cycles négatifs (si détectés)

## Utilisation des Composants Individuels {#using-individual-components}

### 1. Création d'un Graphe

Pour créer un graphe, importez la classe `Graphe` et ajoutez des arcs avec leurs coûts :

```python
from complete_code import Graphe

# Créer un graphe avec 5 sommets (0 à 4)
g = Graphe(5)

# Ajouter des arcs dirigés avec des poids
g.ajouter_arc(0, 1, 4.0)  # Arc de 0 à 1 avec coût 4
g.ajouter_arc(0, 2, 2.0)  # Arc de 0 à 2 avec coût 2
g.ajouter_arc(1, 3, 1.0)  # Arc de 1 à 3 avec coût 1
```

**Conseils pour construire un graphe :**
- Utilisez `Graphe(n)` où `n` est le nombre de sommets
- Les sommets sont numérotés de 0 à n-1
- `ajouter_arc(i, j, cout)` ajoute un arc dirigé de i vers j avec le coût spécifié
- Vous pouvez ajouter plusieurs arcs depuis le même sommet

### 2. Exécution de l'Algorithme de Dijkstra

**Prérequis** : Tous les poids d'arcs doivent être non négatifs.

```python
from complete_code import dijkstra, afficher_resultats

# Calculer les chemins les plus courts depuis le sommet source 0
distances, predecesseurs = dijkstra(g, 0)

# Afficher les résultats
afficher_resultats(distances, predecesseurs, 0, g)
```

**Comment utiliser Dijkstra séparément :**
- Importez `dijkstra` et `afficher_resultats`
- Appelez `dijkstra(graphe, source)` pour obtenir distances et prédécesseurs
- Utilisez `afficher_resultats` pour visualiser les résultats

### 3. Exécution de l'Algorithme de Bellman-Ford

**Utilisez quand** : Le graphe peut contenir des poids d'arcs négatifs.

```python
from complete_code import bellman_ford

distances, predecesseurs, cycle_negatif = bellman_ford(g, 0)

if cycle_negatif:
    print("Avertissement : Cycle négatif détecté !")
else:
    afficher_resultats(distances, predecesseurs, 0, g)
```

**Comment utiliser Bellman-Ford séparément :**
- Importez `bellman_ford`
- La fonction retourne aussi un booléen pour détecter les cycles négatifs
- Vérifiez toujours la présence de cycles négatifs avant d'utiliser les résultats

### 4. Exécution de l'Algorithme de Ford

**Utilisez quand** : Version optimisée de Bellman-Ford pour graphes avec poids négatifs.

```python
from complete_code import ford

distances, predecesseurs, cycle_negatif = ford(g, 0)

if cycle_negatif:
    print("Avertissement : Cycle négatif détecté !")
else:
    afficher_resultats(distances, predecesseurs, 0, g)
```

**Comment utiliser Ford séparément :**
- Importez `ford`
- Même interface que Bellman-Ford mais souvent plus performant en pratique

### 5. Exécution de l'Algorithme de Floyd-Warshall

**Utilisez quand** : Besoin des chemins les plus courts entre toutes les paires de sommets.

```python
import numpy as np
from complete_code import floyd_warshall, reconstruire_chemin_floyd

# Créer matrice de coûts (utilisez float('inf') pour aucun arc)
INF = float('inf')
matrice_couts = np.array([
    [0, 4, 2, INF],
    [INF, 0, 1, 3],
    [INF, INF, 0, 5],
    [INF, INF, INF, 0]
])

distances, sommets_suivants = floyd_warshall(matrice_couts)

# Obtenir le chemin de 0 à 3
chemin = reconstruire_chemin_floyd(sommets_suivants, 0, 3)
print(f"Chemin le plus court : {' -> '.join(map(str, chemin))}")
print(f"Distance : {distances[0, 3]}")
```

**Comment utiliser Floyd-Warshall séparément :**
- Importez `floyd_warshall` et `reconstruire_chemin_floyd`
- Préparez une matrice de coûts (diagonale à 0, INF pour arcs manquants)
- Utilisez `reconstruire_chemin_floyd` pour obtenir les chemins spécifiques

### 6. Reconstruction des Chemins

Pour les algorithmes à source unique (Dijkstra, Bellman-Ford, Ford) :

```python
from complete_code import reconstruire_chemin

# Obtenir le chemin de source à cible
chemin = reconstruire_chemin(predecesseurs, source, cible)
if chemin:
    print(f"Chemin : {' -> '.join(map(str, chemin))}")
else:
    print("Aucun chemin n'existe")
```

**Comment reconstruire les chemins :**
- Utilisez `reconstruire_chemin` avec la liste des prédécesseurs
- Retourne une liste vide si aucun chemin n'existe

## Exemples {#examples}

### Exemple 1 : Utilisation Basique de Dijkstra

```python
from complete_code import Graphe, dijkstra, afficher_resultats

# Créer un graphe simple
g = Graphe(4)
g.ajouter_arc(0, 1, 1)
g.ajouter_arc(0, 2, 4)
g.ajouter_arc(1, 2, 2)
g.ajouter_arc(1, 3, 5)
g.ajouter_arc(2, 3, 1)

# Exécuter Dijkstra depuis le sommet 0
distances, predecesseurs = dijkstra(g, 0)
afficher_resultats(distances, predecesseurs, 0, g)
```

### Exemple 2 : Gestion des Poids Négatifs

```python
from complete_code import Graphe, bellman_ford, afficher_resultats

# Créer un graphe avec poids négatifs
g = Graphe(4)
g.ajouter_arc(0, 1, 4)
g.ajouter_arc(0, 2, 2)
g.ajouter_arc(1, 2, -1)  # Poids négatif
g.ajouter_arc(2, 3, 1)

# Utiliser Bellman-Ford
distances, predecesseurs, cycle_negatif = bellman_ford(g, 0)

if cycle_negatif:
    print("Le graphe contient un cycle négatif")
else:
    afficher_resultats(distances, predecesseurs, 0, g)
```

### Exemple 3 : Chemins les Plus Courts Toutes Paires

```python
import numpy as np
from complete_code import floyd_warshall

# Définir matrice de coûts
couts = np.array([
    [0, 3, 8, float('inf')],
    [float('inf'), 0, float('inf'), 1],
    [float('inf'), 4, 0, float('inf')],
    [2, float('inf'), -5, 0]
])

distances, sommets_suivants = floyd_warshall(couts)

print("Distances toutes paires :")
print(distances)
```

## Interprétation des Sorties {#output-interpretation}

### Valeurs de Distance
- **Nombre fini** : Distance du chemin le plus court depuis la source
- **`inf`** : Sommet inaccessible depuis la source
- **Nombre négatif** : Possible avec poids négatifs (Bellman-Ford/Ford)

### Reconstruction des Chemins
- **Chemin valide** : Liste de sommets de la source à la destination
- **Liste vide** : Aucun chemin n'existe entre les sommets

### Messages Spécifiques aux Algorithmes
- **"Circuit absorbant détecté"** : Cycle négatif trouvé (Bellman-Ford/Ford)
- **Affichage de la structure du graphe** : Montre tous les arcs et leurs poids

### Notes de Performance
- **Dijkstra** : Le plus rapide pour poids positifs, O((n+m) log n) avec tas
- **Bellman-Ford** : Gère les négatifs, O(n×m), détecte les cycles négatifs
- **Ford** : Bellman-Ford optimisé, meilleures performances moyennes
- **Floyd-Warshall** : Toutes paires, O(n³), bon pour graphes denses

## Dépannage {#troubleshooting}

### Problèmes Courants

1. **Erreur d'Importation** : Assurez-vous que `complete_code.py` est dans votre chemin Python
2. **NumPy Non Trouvé** : Installez avec `pip install numpy`
3. **Boucle Infinie** : Vérifiez les cycles négatifs dans Bellman-Ford/Ford
4. **Résultats Incorrects** : Vérifiez la construction du graphe et les directions des arcs

### Guide de Sélection d'Algorithme

| Scénario | Algorithme Recommandé |
|----------|----------------------|
| Poids positifs uniquement | Dijkstra |
| Peut avoir des poids négatifs | Bellman-Ford ou Ford |
| Besoin de chemins les plus courts toutes paires | Floyd-Warshall |
| Grand graphe clairsemé | Dijkstra |
| Graphe dense | Floyd-Warshall |
| Doit détecter les cycles négatifs | Bellman-Ford ou Ford |

### Conseils de Performance

- Utilisez Dijkstra quand possible (le plus rapide pour poids positifs)
- Pour poids négatifs, Ford fonctionne souvent mieux que Bellman-Ford
- Floyd-Warshall est idéal quand vous avez besoin de toutes paires (pas seulement source unique)
- Considérez la taille du graphe : Floyd-Warshall peut être lent pour n > 1000

## Conclusion

Cette implémentation fournit une boîte à outils complète pour les problèmes de chemins les plus courts. Commencez par la fonction de test pour comprendre les algorithmes, puis adaptez les fonctions individuelles pour vos cas d'usage spécifiques. N'oubliez pas de choisir l'algorithme approprié basé sur les caractéristiques de votre graphe (poids, densité, exigences).
