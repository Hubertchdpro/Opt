"""
Complete Python Implementation of Shortest Path Algorithms
Extracted from the comprehensive report on graph algorithms.

This file contains all the code implementations for:
- Graph data structure
- Dijkstra's algorithm
- Bellman-Ford algorithm (with Ford optimization)
- Matrix-based methods (Floyd-Warshall)
- Utility functions and test examples
"""

from typing import List, Dict, Tuple, Set
from collections import defaultdict, deque
import heapq
import numpy as np


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


def ford(graphe: Graphe, s: int) -> Tuple[List[float], List[int], bool]:
    """
    Algorithme de Ford (variante optimisée de Bellman-Ford).

    Args:
        graphe: Le graphe valué
        s: Sommet source

    Returns:
        lambda_vals: Liste des valeurs λi
        predecesseurs: Liste des prédécesseurs
        circuit_absorbant: True si circuit absorbant détecté
    """
    # L'implémentation de Bellman-Ford ci-dessus utilise déjà l'approche de Ford
    # avec une file d'attente pour optimiser les relaxations
    return bellman_ford(graphe, s)


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
