# Rapport sur les Algorithmes de Chemins Minimaux dans les Graphes

## Table des matières

1. [Introduction aux graphes et problématique](https://claude.ai/chat/5c31b775-dbfb-4cf4-aaf7-b6842ac809ec#introduction)
2. [Terminologie et notations](https://claude.ai/chat/5c31b775-dbfb-4cf4-aaf7-b6842ac809ec#terminologie)
3. [Algorithme de Dijkstra](https://claude.ai/chat/5c31b775-dbfb-4cf4-aaf7-b6842ac809ec#dijkstra)
4. [Algorithme de Bellman-Ford](https://claude.ai/chat/5c31b775-dbfb-4cf4-aaf7-b6842ac809ec#bellman)
5. [Algorithme de Ford](https://claude.ai/chat/5c31b775-dbfb-4cf4-aaf7-b6842ac809ec#ford)
6. [Méthode matricielle](https://claude.ai/chat/5c31b775-dbfb-4cf4-aaf7-b6842ac809ec#matricielle)
7. [Analyse comparative](https://claude.ai/chat/5c31b775-dbfb-4cf4-aaf7-b6842ac809ec#comparaison)
8. [Implémentations Python](https://claude.ai/chat/5c31b775-dbfb-4cf4-aaf7-b6842ac809ec#implementations)
9. [Conclusion](https://claude.ai/chat/5c31b775-dbfb-4cf4-aaf7-b6842ac809ec#conclusion)

---

## 1. Introduction aux graphes et problématique {#introduction}

### 1.1 Contexte

Les graphes constituent une structure mathématique fondamentale pour modéliser des relations entre objets. Un **graphe valué** (ou graphe pondéré) est un graphe dont les arcs (ou arêtes) possèdent une valeur numérique appelée **coût** ou **valuation**.

### 1.2 Définitions de base

**Graphe valué** : Un graphe G = (V, E, C) où :

- V = {v₀, v₁, ..., vₙ₋₁} est l'ensemble des sommets
- E ⊆ V × V est l'ensemble des arcs
- C : E → ℝ est la fonction de coût (ou valuation)

**Notation** : Pour un arc (vᵢ, vⱼ) ∈ E, on note Cᵢⱼ son coût. Si (vᵢ, vⱼ) ∉ E, on pose Cᵢⱼ = ∞.

**Chemin** : Une séquence de sommets μ = [s = v₀, v₁, ..., vₚ = vₜ] où chaque paire consécutive forme un arc.

**Valeur d'un chemin** : La somme des coûts des arcs qui le composent. Pour un chemin μ allant de s à vₜ :

```
Valeur(μ) = Σ Cᵢⱼ pour tous les arcs (vᵢ, vⱼ) du chemin
```

**Chemin minimal** : Un chemin de valeur minimale entre deux sommets donnés.

### 1.3 Problématique

Le problème du chemin minimal consiste à déterminer, pour un sommet source s et un sommet destination vₜ dans un graphe valué, le chemin de coût total minimal reliant s à vₜ.

**Notation importante** : On note λᵢ la valeur minimale d'un chemin allant du sommet source s au sommet vᵢ.

---

## 2. Terminologie et notations {#terminologie}

### 2.1 Notations utilisées

Conformément aux images fournies, nous utilisons les notations suivantes :

- **n** = |V| : nombre de sommets du graphe
- **m** = |E| : nombre d'arcs du graphe
- **s** : sommet source (origine)
- **vₜ** : sommet destination (cible)
- **λᵢ** : valeur minimale du chemin de s à vᵢ
- **Cᵢⱼ** : coût de l'arc (vᵢ, vⱼ)
- **S** : ensemble des sommets traités/visités
- **E\S** : ensemble des sommets non encore traités

### 2.2 Matrice de valuation

La matrice C° = (Cᵢⱼ) représente les coûts directs entre sommets :

- Cᵢⱼ = coût de l'arc si (vᵢ, vⱼ) ∈ E
- Cᵢⱼ = ∞ si (vᵢ, vⱼ) ∉ E

Pour les graphes sans circuit absorbant, cette représentation matricielle est fondamentale.

### 2.3 Arbre couvrant de chemin minimal

Un **arbre couvrant de chemin minimal** est un sous-graphe du graphe original qui :

- Contient tous les sommets
- Forme un arbre (connexe et sans cycle)
- Pour chaque sommet, le chemin depuis la source dans cet arbre est un chemin minimal

---

## 3. Algorithme de Dijkstra {#dijkstra}

### 3.1 Principe de l'algorithme

L'algorithme de Dijkstra s'applique aux graphes dont **toutes les valuations d'arcs sont non négatives**. Cette méthode minimise le nombre d'ajustements des valeurs λᵢ.

**Idée centrale** : À chaque itération, on choisit le sommet non traité ayant la plus petite valeur λᵢ actuelle, et on le marque comme définitivement traité.

### 3.2 Description formelle

**Initialisation** :

1. S = {s}, λ₀ = 0
2. λᵢ = C₀ᵢ si (s, vᵢ) ∈ V
3. λᵢ = ∞ sinon

**Itération** : Tant que S ≠ E (ensemble de tous les sommets)

(i) Choisir vₖ ∈ E\S tel que λₖ = min{λᵢ : vᵢ ∈ E\S}

(ii) S = S ∪ {vₖ}

(iii) Pour tout vⱼ ∈ E\S, successeur de vₖ, faire :

```
λⱼ = min{λⱼ, λₖ + Cₖⱼ}
```

### 3.3 Justification théorique

**Propriété fondamentale** : Si toutes les valuations sont non négatives, alors lorsqu'un sommet vₖ est ajouté à S avec la valeur λₖ minimale parmi E\S, cette valeur λₖ est définitive et représente bien la distance minimale de s à vₖ.

**Preuve** : Supposons par l'absurde qu'il existe un chemin de s à vₖ de valeur strictement inférieure à λₖ. Ce chemin doit nécessairement sortir de S à un moment donné via un sommet vᵢ ∈ S vers un sommet vⱼ ∉ S. Mais puisque toutes les valuations sont ≥ 0, on aurait λⱼ ≤ λᵢ + Cᵢⱼ ≤ λₖ, ce qui contredit le choix de vₖ comme minimum.

### 3.4 Complexité

**Complexité temporelle** :

- Avec une liste : O(n²)
- Avec un tas binaire : O((n + m) log n)
- Avec un tas de Fibonacci : O(m + n log n)

**Complexité spatiale** : O(n + m) pour stocker le graphe et les distances.

### 3.5 Exemple d'exécution

Considérons le graphe suivant :

```
Graphe :
     2         3
  0 ---> 1 ---> 3
  |      |      ^
 6|     1|      |2
  v      v      |
  2 ---> 4 -----
     5      4
```

**Étapes de Dijkstra depuis le sommet 0** :

|Itération|Sommet choisi|S|λ₀|λ₁|λ₂|λ₃|λ₄|
|---|---|---|---|---|---|---|---|
|Init|0|{0}|0|∞|∞|∞|∞|
|1|0|{0}|0|2|6|∞|∞|
|2|1|{0,1}|0|2|3|5|∞|
|3|2|{0,1,2}|0|2|3|5|8|
|4|3|{0,1,2,3}|0|2|3|5|7|
|5|4|{0,1,2,3,4}|0|2|3|5|7|

**Résultat** :

- Chemin minimal de 0 à 4 : 0 → 1 → 3 → 4
- Valeur : 2 + 3 + 2 = 7

---

## 4. Algorithme de Bellman-Ford {#bellman}

### 4.1 Motivation

L'algorithme de Dijkstra ne fonctionne pas avec des arcs de coût négatif. L'algorithme de **Bellman-Ford** permet de traiter les graphes avec des valuations négatives, à condition qu'il n'y ait pas de **circuit absorbant** (circuit de coût total négatif).

### 4.2 Principe

Au lieu de traiter les sommets dans un ordre particulier, Bellman-Ford effectue des **relaxations** systématiques sur tous les arcs du graphe, et répète ce processus (n-1) fois maximum.

**Relaxation** : Pour un arc (vᵢ, vⱼ), vérifier si passer par vᵢ améliore la distance vers vⱼ :

```
si λⱼ > λᵢ + Cᵢⱼ alors λⱼ = λᵢ + Cᵢⱼ
```

### 4.3 Description formelle

**Initialisation** :

1. λ₀ = 0 (sommet source s = v₀)
2. λᵢ = ∞ pour i ≠ 0

**Itération** : Pour k = 1 à n-1

- Pour chaque arc (vᵢ, vⱼ) ∈ E, faire :

```
λⱼ = min{λⱼ, λᵢ + Cᵢⱼ}
```

**Détection de circuit absorbant** : Après n-1 itérations

- Si on peut encore réduire une valeur λⱼ, alors il existe un circuit absorbant

### 4.4 Justification théorique

**Lemme** : Après k itérations, λᵢ est au plus égal à la valeur du plus court chemin de s à vᵢ utilisant au plus k arcs.

**Preuve par induction** :

- Après 0 itération : vrai (λ₀ = 0, autres infinis)
- Si vrai au rang k, alors au rang k+1, chaque arc (vᵢ, vⱼ) permet de calculer un chemin de k+1 arcs
- Après n-1 itérations, tous les chemins simples (sans cycle) sont considérés

**Détection de circuit absorbant** : Si après n-1 passes une valeur peut encore diminuer, c'est qu'on peut emprunter un cycle qui réduit le coût, donc un circuit absorbant.

### 4.5 Complexité

**Complexité temporelle** : O(n × m)

- n-1 itérations
- À chaque itération, on examine tous les m arcs

**Complexité spatiale** : O(n + m)

### 4.6 Exemple avec coûts négatifs

```
Graphe :
     4         -3
  0 ---> 1 ---> 3
  |      |      
 2|     1|      
  v      v      
  2 <-----------
       -2
```

**Exécution de Bellman-Ford depuis 0** :

|Itération|λ₀|λ₁|λ₂|λ₃|
|---|---|---|---|---|
|Init|0|∞|∞|∞|
|1|0|4|2|∞|
|2|0|0|1|1|
|3|0|0|-1|1|

Le chemin minimal vers 3 : 0 → 1 → 3, valeur = 1

---

## 5. Algorithme de Ford {#ford}

### 4.1 Présentation

L'algorithme de Ford est une **variante optimisée** de l'algorithme de Bellman-Ford. Au lieu de parcourir systématiquement tous les arcs à chaque itération, Ford maintient une liste des sommets dont la valeur λ a été modifiée, et ne traite que leurs successeurs.

### 5.2 Principe d'optimisation

**Observation clé** : Si λᵢ n'a pas changé lors de l'itération précédente, alors examiner à nouveau les arcs sortants de vᵢ est inutile.

**Solution** : Maintenir un ensemble (ou file) des sommets "actifs" dont la distance vient d'être améliorée.

### 5.3 Description formelle

**Initialisation** :

1. λ₀ = 0, λᵢ = ∞ pour i ≠ 0
2. File F = {v₀}

**Itération** : Tant que F ≠ ∅

(i) Extraire un sommet vᵢ de F

(ii) Pour chaque arc (vᵢ, vⱼ) sortant de vᵢ :

```
si λⱼ > λᵢ + Cᵢⱼ alors
    λⱼ = λᵢ + Cᵢⱼ
    si vⱼ ∉ F alors ajouter vⱼ à F
```

(iii) Répéter l'étape (ii) tant que λᵢ n'a pas été modifié dans (ii)

### 5.4 Cas particulier : graphes sans circuit

Pour les graphes sans circuit (DAG - Directed Acyclic Graph), l'algorithme de Ford est particulièrement efficace et peut même être simplifié en effectuant un parcours en ordre topologique.

### 5.5 Complexité

**Dans le pire cas** : O(n × m) comme Bellman-Ford

**En pratique** : Souvent bien meilleur car on évite de traiter des arcs inutiles.

**Cas optimal** (graphe sans circuit) : O(n + m)

### 5.6 Exemple d'exécution

Reprenons le graphe de la section précédente :

```
     4         -3
  0 ---> 1 ---> 3
  |      |      
 2|     1|      
  v      v      
  2 <-----------
       -2
```

**Exécution de Ford depuis 0** :

|Étape|Sommet traité|F (file)|λ₀|λ₁|λ₂|λ₃|
|---|---|---|---|---|---|---|
|Init|-|{0}|0|∞|∞|∞|
|1|0|{1,2}|0|4|2|∞|
|2|1|{2,3}|0|4|2|1|
|3|2|{3,1}|0|0|2|1|
|4|3|{1}|0|0|2|1|
|5|1|{2}|0|0|1|1|
|6|2|{}|0|0|1|1|

La file devient vide, l'algorithme se termine.

---

## 6. Méthode matricielle {#matricielle}

### 6.1 Approche par matrices

La méthode matricielle ne peut être appliquée que pour les **graphes sans circuit absorbant**. Elle utilise les puissances successives de la matrice de valuation.

### 6.2 Principe

Soit C° = (Cᵢⱼ) la matrice des coûts initiaux. On définit :

**C¹** : matrice des chemins de longueur ≤ 1 **C²** : matrice des chemins de longueur ≤ 2 **Cᵏ** : matrice des chemins de longueur ≤ k

La relation de récurrence est :

```
Cᵢⱼᵏ = min{Cᵢⱼᵏ⁻¹, min{Cᵢₗᵏ⁻¹ + Cₗⱼ}}
```

Autrement dit, pour aller de vᵢ à vⱼ avec au plus k arcs, soit on utilise au plus k-1 arcs, soit on passe par un intermédiaire vₗ.

### 6.3 Algorithme

**Initialisation** : C° avec Cᵢⱼ° = coût de (vᵢ, vⱼ) ou ∞

**Itération** : Pour k = 1 à n-1

```
Pour i = 0 à n-1
    Pour j = 0 à n-1
        Cᵢⱼᵏ = min{Cᵢⱼᵏ⁻¹, min_l{Cᵢₗᵏ⁻¹ + Cₗⱼ}}
```

**Résultat** : Après n-1 itérations, Cⁿ⁻¹ contient les distances minimales.

### 6.4 Variante : Algorithme de Floyd-Warshall

Une amélioration consiste à calculer les chemins minimaux entre **toutes les paires** de sommets en utilisant une approche de programmation dynamique basée sur les sommets intermédiaires :

```
Pour k = 0 à n-1
    Pour i = 0 à n-1
        Pour j = 0 à n-1
            Dᵢⱼ = min{Dᵢⱼ, Dᵢₖ + Dₖⱼ}
```

### 6.5 Complexité

**Méthode matricielle simple** : O(n⁴)

**Floyd-Warshall** : O(n³)

### 6.6 Exemple

Soit le graphe représenté par la matrice initiale C° :

```
     v₀  v₁  v₂  v₃
v₀ [  0   4   2   ∞ ]
v₁ [  ∞   0   1   3 ]
v₂ [  ∞   ∞   0   5 ]
v₃ [  ∞   ∞   ∞   0 ]
```

**Application de Floyd-Warshall** :

Après traitement de k=0 (passage par v₀) :

```
     v₀  v₁  v₂  v₃
v₀ [  0   4   2   ∞ ]
v₁ [  ∞   0   1   3 ]
v₂ [  ∞   ∞   0   5 ]
v₃ [  ∞   ∞   ∞   0 ]
```

Après k=1 (passage par v₁) :

```
     v₀  v₁  v₂  v₃
v₀ [  0   4   2   7 ]
v₁ [  ∞   0   1   3 ]
v₂ [  ∞   ∞   0   5 ]
v₃ [  ∞   ∞   ∞   0 ]
```

Après k=2 :

```
     v₀  v₁  v₂  v₃
v₀ [  0   4   2   7 ]
v₁ [  ∞   0   1   3 ]
v₂ [  ∞   ∞   0   5 ]
v₃ [  ∞   ∞   ∞   0 ]
```

Après k=3 : pas de changement.

**Résultat final** : Distance minimale de v₀ à v₃ = 7

---

## 7. Analyse comparative {#comparaison}

### 7.1 Tableau comparatif

|Algorithme|Coûts négatifs|Circuit absorbant|Complexité|Cas d'usage optimal|
|---|---|---|---|---|
|**Dijkstra**|Non|N/A|O(n²) ou O((n+m)log n)|Graphes denses avec coûts ≥ 0|
|**Bellman-Ford**|Oui|Détecté|O(n×m)|Graphes avec coûts négatifs|
|**Ford**|Oui|Détecté|O(n×m) pire cas, meilleur en pratique|Optimisation de Bellman-Ford|
|**Matricielle**|Oui|Non|O(n⁴)|Pédagogique, petits graphes|
|**Floyd-Warshall**|Oui|Détecté|O(n³)|Tous les chemins minimaux|

### 7.2 Choix de l'algorithme

**Dijkstra** est préférable quand :

- Tous les coûts sont positifs
- On cherche les chemins depuis une seule source
- Le graphe est grand

**Bellman-Ford/Ford** sont nécessaires quand :

- Des coûts négatifs existent
- On doit détecter les circuits absorbants
- Le graphe n'est pas trop dense

**Floyd-Warshall** convient quand :

- On veut tous les chemins minimaux
- Le graphe est petit (n ≤ 500)

### 7.3 Propriétés des chemins minimaux

**Sous-structure optimale** : Un sous-chemin d'un chemin minimal est lui-même minimal.

**Inégalité triangulaire** : λₖ ≤ λᵢ + Cᵢₖ pour tout arc (vᵢ, vₖ).

**Arbre des chemins minimaux** : Les chemins minimaux depuis une source forment un arbre couvrant.

---

## 8. Implémentations Python {#implementations}

### 8.1 Structure de données pour les graphes

```python
from typing import List, Dict, Tuple, Set
from collections import defaultdict, deque
import heapq

class Graphe:
    """
    Représentation d'un graphe valué orienté.
    """
    def __init__(self, n: int):
        """
        Initialise un graphe avec n sommets numérotés de 0 à n-1.
        
        Args:
            n: Nombre de sommets
        """
        self.n = n  # Nombre de sommets
        self.arcs: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        # arcs[i] = [(j, Cij), ...] : liste des successeurs de i avec coûts
    
    def ajouter_arc(self, i: int, j: int, cout: float):
        """
        Ajoute un arc (vi, vj) de coût Cij.
        
        Args:
            i: Sommet origine
            j: Sommet destination
            cout: Coût Cij de l'arc
        """
        self.arcs[i].append((j, cout))
    
    def obtenir_successeurs(self, i: int) -> List[Tuple[int, float]]:
        """
        Retourne la liste des successeurs du sommet i.
        
        Returns:
            Liste de tuples (j, Cij)
        """
        return self.arcs[i]
    
    def obtenir_tous_arcs(self) -> List[Tuple[int, int, float]]:
        """
        Retourne tous les arcs du graphe.
        
        Returns:
            Liste de tuples (i, j, Cij)
        """
        arcs_liste = []
        for i in range(self.n):
            for j, cout in self.arcs[i]:
                arcs_liste.append((i, j, cout))
        return arcs_liste
    
    def afficher(self):
        """Affiche la structure du graphe."""
        print(f"Graphe avec {self.n} sommets:")
        for i in range(self.n):
            if self.arcs[i]:
                print(f"  v{i} → {[(f'v{j}', c) for j, c in self.arcs[i]]}")
```

### 8.2 Implémentation de Dijkstra

```python
def dijkstra(graphe: Graphe, s: int) -> Tuple[List[float], List[int]]:
    """
    Algorithme de Dijkstra pour calculer les chemins minimaux depuis s.
    
    Précondition: Toutes les valuations Cij ≥ 0
    
    Args:
        graphe: Le graphe valué
        s: Sommet source (v0)
    
    Returns:
        lambda_vals: Liste des valeurs λi (distances minimales)
        predecesseurs: Liste des prédécesseurs pour reconstruire les chemins
    """
    n = graphe.n
    
    # Initialisation
    lambda_vals = [float('inf')] * n
    lambda_vals[s] = 0
    predecesseurs = [-1] * n
    
    # Ensemble S des sommets traités (True si traité)
    traites = [False] * n
    
    # File de priorité: (λi, vi)
    file_priorite = [(0, s)]
    
    while file_priorite:
        # (i) Choisir vk ∈ E\S avec λk minimal
        lambda_k, k = heapq.heappop(file_priorite)
        
        # Si déjà traité, passer
        if traites[k]:
            continue
        
        # (ii) S = S ∪ {vk}
        traites[k] = True
        
        # (iii) Pour tout vj successeur de vk dans E\S
        for j, c_kj in graphe.obtenir_successeurs(k):
            if not traites[j]:
                # Relaxation: λj = min{λj, λk + Ckj}
                nouvelle_valeur = lambda_vals[k] + c_kj
                if nouvelle_valeur < lambda_vals[j]:
                    lambda_vals[j] = nouvelle_valeur
                    predecesseurs[j] = k
                    heapq.heappush(file_priorite, (nouvelle_valeur, j))
    
    return lambda_vals, predecesseurs


def reconstruire_chemin(predecesseurs: List[int], s: int, t: int) -> List[int]:
    """
    Reconstruit le chemin minimal de s à t.
    
    Args:
        predecesseurs: Liste des prédécesseurs
        s: Sommet source
        t: Sommet destination
    
    Returns:
        Chemin [s, ..., t] ou [] si aucun chemin
    """
    if predecesseurs[t] == -1 and t != s:
        return []  # Pas de chemin
    
    chemin = []
    courant = t
    while courant != -1:
        chemin.append(courant)
        if courant == s:
            break
        courant = predecesseurs[courant]
    
    return chemin[::-1]  # Inverser pour avoir s → t
```

### 8.3 Implémentation de Bellman-Ford

```python
def bellman_ford(graphe: Graphe, s: int) -> Tuple[List[float], List[int], bool]:
    """
    Algorithme de Bellman-Ford pour graphes avec coûts négatifs.
    
    Args:
        graphe: Le graphe valué
        s: Sommet source
    
    Returns:
        lambda_vals: Liste des valeurs λi
        predecesseurs: Liste des prédécesseurs
        circuit_absorbant: True si circuit absorbant détecté
    """
    n = graphe.n
    
    # Initialisation
    lambda_vals = [float('inf')] * n
    lambda_vals[s] = 0
    predecesseurs = [-1] * n
    
    # File des sommets actifs
    file = deque([s])
    dans_file = {s}
    
    # Compteur de passages dans la file (pour détecter circuits absorbants)
    compteur_passages = [0] * n
    compteur_passages[s] = 1
    
    while file:
        # Extraire un sommet vi de la file
        i = file.popleft()
        dans_file.discard(i)
        
        # Pour chaque arc (vi, vj) sortant de vi
        for j, c_ij in graphe.obtenir_successeurs(i):
            # Relaxation
            if lambda_vals[i] != float('inf'):
                nouvelle_valeur = lambda_vals[i] + c_ij
                if nouvelle_valeur < lambda_vals[j]:
                    lambda_vals[j] = nouvelle_valeur
                    predecesseurs[j] = i
                    
                    # Si vj n'est pas dans la file, l'ajouter
                    if j not in dans_file:
                        file.append(j)
                        dans_file.add(j)
                        compteur_passages[j] += 1
                        
                        # Si un sommet passe plus de n fois, circuit absorbant
                        if compteur_passages[j] > n:
                            return lambda_vals, predecesseurs, True
    
    return lambda_vals, predecesseurs, False
```

### 8.5 Implémentation de la méthode matricielle

```python
import numpy as np

def methode_matricielle(matrice_couts: np.ndarray) -> np.ndarray:
    """
    Calcule les chemins minimaux par la méthode matricielle.
    
    Précondition: Graphe sans circuit absorbant
    
    Args:
        matrice_couts: Matrice C° des coûts (np.inf pour arcs inexistants)
    
    Returns:
        Matrice C^(n-1) des distances minimales
    """
    n = matrice_couts.shape[0]
    C = matrice_couts.copy()
    
    # n-1 itérations
    for k in range(1, n):
        C_new = C.copy()
        
        for i in range(n):
            for j in range(n):
                # C[i,j]^k = min{C[i,j]^(k-1), min_l{C[i,l]^(k-1) + C[l,j]}}
                for l in range(n):
                    C_new[i, j] = min(C_new[i, j], C[i, l] + matrice_couts[l, j])
        
        C = C_new
    
    return C


def floyd_warshall(matrice_couts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithme de Floyd-Warshall pour tous les chemins minimaux.
    
    Args:
        matrice_couts: Matrice des coûts initiaux
    
    Returns:
        distances: Matrice des distances minimales
        suivants: Matrice pour reconstruire les chemins
    """
    n = matrice_couts.shape[0]
    distances = matrice_couts.copy()
    
    # Matrice des sommets suivants pour reconstruction
    suivants = np.full((n, n), -1, dtype=int)
    for i in range(n):
        for j in range(n):
            if matrice_couts[i, j] < float('inf') and i != j:
                suivants[i, j] = j
    
    # Algorithme principal
    for k in range(n):
        for i in range(n):
            for j in range(n):
                nouvelle_distance = distances[i, k] + distances[k, j]
                if nouvelle_distance < distances[i, j]:
                    distances[i, j] = nouvelle_distance
                    suivants[i, j] = suivants[i, k]
    
    return distances, suivants


def reconstruire_chemin_floyd(suivants: np.ndarray, i: int, j: int) -> List[int]:
    """
    Reconstruit le chemin de i à j avec Floyd-Warshall.
    
    Args:
        suivants: Matrice des sommets suivants
        i: Sommet source
        j: Sommet destination
    
    Returns:
        Chemin [i, ..., j]
    """
    if suivants[i, j] == -1:
        return []
    
    chemin = [i]
    while i != j:
        i = suivants[i, j]
        chemin.append(i)
    
    return chemin
```

### 8.6 Fonctions utilitaires et tests

```python
def afficher_resultats(lambda_vals: List[float], predecesseurs: List[int], 
                       s: int, graphe: Graphe):
    """
    Affiche les résultats d'un algorithme de chemin minimal.
    """
    print(f"\nRésultats depuis le sommet source v{s}:")
    print("-" * 50)
    
    for i in range(graphe.n):
        if lambda_vals[i] == float('inf'):
            print(f"  v{i}: distance = ∞ (inaccessible)")
        else:
            chemin = reconstruire_chemin(predecesseurs, s, i)
            chemin_str = " → ".join([f"v{v}" for v in chemin])
            print(f"  v{i}: λ{i} = {lambda_vals[i]:.1f}, chemin: {chemin_str}")


def creer_graphe_exemple_1() -> Graphe:
    """
    Crée le graphe d'exemple de la Figure 4.4 (Dijkstra).
    """
    g = Graphe(5)  # 5 sommets: 0, 1, 2, 3, 4
    
    # Arcs avec coûts positifs
    g.ajouter_arc(0, 1, 4)
    g.ajouter_arc(0, 2, 1)
    g.ajouter_arc(1, 4, 4)
    g.ajouter_arc(2, 1, 1)
    g.ajouter_arc(2, 3, 5)
    g.ajouter_arc(3, 4, 3)
    
    return g


def creer_graphe_exemple_2() -> Graphe:
    """
    Crée un graphe avec coûts négatifs pour Bellman-Ford.
    """
    g = Graphe(4)  # 4 sommets
    
    g.ajouter_arc(0, 1, 4)
    g.ajouter_arc(0, 2, 2)
    g.ajouter_arc(1, 2, 1)
    g.ajouter_arc(1, 3, -3)
    g.ajouter_arc(2, 1, -2)
    
    return g


def tester_algorithmes():
    """
    Teste les différents algorithmes sur des exemples.
    """
    print("=" * 70)
    print("TEST 1: Dijkstra sur graphe avec coûts positifs")
    print("=" * 70)
    
    g1 = creer_graphe_exemple_1()
    g1.afficher()
    
    lambda_vals, predecesseurs = dijkstra(g1, 0)
    afficher_resultats(lambda_vals, predecesseurs, 0, g1)
    
    print("\n" + "=" * 70)
    print("TEST 2: Bellman-Ford sur graphe avec coûts négatifs")
    print("=" * 70)
    
    g2 = creer_graphe_exemple_2()
    g2.afficher()
    
    lambda_vals, predecesseurs, circuit = bellman_ford(g2, 0)
    
    if circuit:
        print("\n⚠️  ATTENTION: Circuit absorbant détecté!")
    else:
        afficher_resultats(lambda_vals, predecesseurs, 0, g2)
    
    print("\n" + "=" * 70)
    print("TEST 3: Ford sur le même graphe")
    print("=" * 70)
    
    lambda_vals, predecesseurs, circuit = ford(g2, 0)
    
    if circuit:
        print("\n⚠️  ATTENTION: Circuit absorbant détecté!")
    else:
        afficher_resultats(lambda_vals, predecesseurs, 0, g2)
    
    print("\n" + "=" * 70)
    print("TEST 4: Floyd-Warshall")
    print("=" * 70)
    
    # Créer matrice de coûts
    INF = float('inf')
    matrice = np.array([
        [0, 4, 2, INF],
        [INF, 0, 1, -3],
        [INF, -2, 0, INF],
        [INF, INF, INF, 0]
    ])
    
    distances, suivants = floyd_warshall(matrice)
    
    print("\nMatrice des distances minimales:")
    print(distances)
    
    print("\nChemins minimaux:")
    for i in range(4):
        for j in range(4):
            if i != j and distances[i, j] < INF:
                chemin = reconstruire_chemin_floyd(suivants, i, j)
                chemin_str = " → ".join([f"v{v}" for v in chemin])
                print(f"  v{i} → v{j}: distance = {distances[i,j]:.1f}, "
                      f"chemin: {chemin_str}")


if __name__ == "__main__":
    tester_algorithmes()
```

---

## 9. Conclusion {#conclusion}

### 9.1 Synthèse

Les algorithmes de chemins minimaux constituent un domaine fondamental de la théorie des graphes avec de nombreuses applications pratiques : routage réseau, GPS, logistique, ordonnancement, etc.

**Points clés** :

1. **Dijkstra** : Optimal pour graphes avec coûts positifs, complexité O(n²) ou O((n+m) log n)
    
2. **Bellman-Ford** : Gère les coûts négatifs et détecte les circuits absorbants, complexité O(n×m)
    
3. **Ford** : Optimisation de Bellman-Ford, même complexité théorique mais meilleure en pratique
    
4. **Méthodes matricielles** : Approche différente, utile pour tous les chemins (Floyd-Warshall en O(n³))
    

### 9.2 Critères de choix

|Critère|Algorithme recommandé|
|---|---|
|Coûts tous positifs|Dijkstra|
|Coûts négatifs possibles|Bellman-Ford ou Ford|
|Tous les chemins minimaux|Floyd-Warshall|
|Graphe très dense|Dijkstra avec liste|
|Graphe très creux|Dijkstra avec tas|
|Détection circuit absorbant|Bellman-Ford ou Ford|

### 9.3 Extensions possibles

- **A*** : Dijkstra avec heuristique pour recherche guidée
- **Bidirectionnel** : Recherche simultanée depuis source et destination
- **Chemins k plus courts** : Variantes pour trouver les k meilleurs chemins
- **Graphes dynamiques** : Algorithmes incrémentaux pour graphes changeants

### 9.4 Applications pratiques

**Transport et logistique** :

- Calcul d'itinéraires GPS
- Optimisation de tournées de livraison
- Planification de trajets multimodaux

**Réseaux informatiques** :

- Protocoles de routage (OSPF, RIP)
- Optimisation de flux de données
- Gestion de la qualité de service

**Finance** :

- Arbitrage sur les marchés de change
- Optimisation de portefeuilles
- Évaluation de risques

**Autres domaines** :

- Planification robotique
- Analyse de réseaux sociaux
- Bioinformatique (alignement de séquences)

### 9.5 Conclusion finale

La maîtrise de ces algorithmes classiques est essentielle pour tout informaticien. Leur compréhension approfondie permet non seulement de les appliquer efficacement, mais aussi de les adapter à des problèmes spécifiques et de concevoir de nouveaux algorithmes pour des variantes plus complexes.

Les implémentations fournies dans ce rapport constituent une base solide pour l'expérimentation et l'application pratique. Il est recommandé de les tester sur différents types de graphes et de mesurer leurs performances pour bien comprendre leur comportement dans diverses situations.

---

## Bibliographie

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). _Introduction to Algorithms_ (3rd ed.). MIT Press.
    
2. Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. _Numerische Mathematik_, 1(1), 269-271.
    
3. Bellman, R. (1958). On a routing problem. _Quarterly of Applied Mathematics_, 16(1), 87-90.
    
4. Ford, L. R., & Fulkerson, D. R. (1962). _Flows in Networks_. Princeton University Press.
    
5. Floyd, R. W. (1962). Algorithm 97: Shortest path. _Communications of the ACM_, 5(6), 345.
    
6. Warshall, S. (1962). A theorem on boolean matrices. _Journal of the ACM_, 9(1), 11-12.
    
