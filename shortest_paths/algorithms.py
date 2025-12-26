from typing import List, Tuple
from collections import deque
import heapq
import numpy as np

from .graph import Graphe


def dijkstra(graphe: Graphe, s: int) -> Tuple[List[float], List[int]]:
    """Algorithme de Dijkstra pour calculer les chemins minimaux depuis s.

    Précondition: Toutes les valuations Cij ≥ 0
    """
    n = graphe.n

    lambda_vals = [float('inf')] * n
    lambda_vals[s] = 0
    predecesseurs = [-1] * n

    traites = [False] * n
    file_priorite = [(0, s)]

    while file_priorite:
        lambda_k, k = heapq.heappop(file_priorite)
        if traites[k]:
            continue
        traites[k] = True

        for j, c_kj in graphe.obtenir_successeurs(k):
            if not traites[j]:
                nouvelle_valeur = lambda_vals[k] + c_kj
                if nouvelle_valeur < lambda_vals[j]:
                    lambda_vals[j] = nouvelle_valeur
                    predecesseurs[j] = k
                    heapq.heappush(file_priorite, (nouvelle_valeur, j))

    return lambda_vals, predecesseurs


def reconstruire_chemin(predecesseurs: List[int], s: int, t: int) -> List[int]:
    if predecesseurs[t] == -1 and t != s:
        return []
    chemin = []
    courant = t
    while courant != -1:
        chemin.append(courant)
        if courant == s:
            break
        courant = predecesseurs[courant]
    return chemin[::-1]


def bellman_ford(graphe: Graphe, s: int) -> Tuple[List[float], List[int], bool]:
    """Version classique de Bellman-Ford (O(n*m))."""
    n = graphe.n
    lambda_vals = [float('inf')] * n
    lambda_vals[s] = 0
    predecesseurs = [-1] * n

    arcs = graphe.obtenir_tous_arcs()
    for _ in range(n - 1):
        updated = False
        for i, j, c in arcs:
            if lambda_vals[i] != float('inf') and lambda_vals[i] + c < lambda_vals[j]:
                lambda_vals[j] = lambda_vals[i] + c
                predecesseurs[j] = i
                updated = True
        if not updated:
            break

    # Détection de circuit absorbant
    for i, j, c in arcs:
        if lambda_vals[i] != float('inf') and lambda_vals[i] + c < lambda_vals[j]:
            return lambda_vals, predecesseurs, True

    return lambda_vals, predecesseurs, False


def ford(graphe: Graphe, s: int) -> Tuple[List[float], List[int], bool]:
    """Variante optimisée (SPFA-like) nommée Ford dans le document."""
    n = graphe.n
    lambda_vals = [float('inf')] * n
    lambda_vals[s] = 0
    predecesseurs = [-1] * n

    file = deque([s])
    dans_file = {s}
    compteur_passages = [0] * n
    compteur_passages[s] = 1

    while file:
        i = file.popleft()
        dans_file.discard(i)

        for j, c_ij in graphe.obtenir_successeurs(i):
            if lambda_vals[i] != float('inf') and lambda_vals[i] + c_ij < lambda_vals[j]:
                lambda_vals[j] = lambda_vals[i] + c_ij
                predecesseurs[j] = i
                if j not in dans_file:
                    file.append(j)
                    dans_file.add(j)
                    compteur_passages[j] += 1
                    if compteur_passages[j] > n:
                        return lambda_vals, predecesseurs, True

    return lambda_vals, predecesseurs, False


def methode_matricielle(matrice_couts: np.ndarray) -> np.ndarray:
    n = matrice_couts.shape[0]
    C = matrice_couts.copy()

    for k in range(1, n):
        C_new = C.copy()
        for i in range(n):
            for j in range(n):
                for l in range(n):
                    C_new[i, j] = min(C_new[i, j], C[i, l] + matrice_couts[l, j])
        C = C_new

    return C


def floyd_warshall(matrice_couts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = matrice_couts.shape[0]
    distances = matrice_couts.copy()
    suivants = np.full((n, n), -1, dtype=int)
    for i in range(n):
        for j in range(n):
            if matrice_couts[i, j] < float('inf') and i != j:
                suivants[i, j] = j

    for k in range(n):
        for i in range(n):
            for j in range(n):
                nouvelle_distance = distances[i, k] + distances[k, j]
                if nouvelle_distance < distances[i, j]:
                    distances[i, j] = nouvelle_distance
                    suivants[i, j] = suivants[i, k]

    return distances, suivants


def reconstruire_chemin_floyd(suivants: np.ndarray, i: int, j: int) -> List[int]:
    if suivants[i, j] == -1:
        return []
    chemin = [i]
    while i != j:
        i = suivants[i, j]
        chemin.append(i)
    return chemin
