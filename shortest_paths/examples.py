from typing import List
import numpy as np

from .graph import Graphe
from .algorithms import (
    dijkstra,
    bellman_ford,
    ford,
    floyd_warshall,
    reconstruire_chemin,
    reconstruire_chemin_floyd,
)


def afficher_resultats(lambda_vals: List[float], predecesseurs: List[int], s: int, graphe: Graphe):
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
    g = Graphe(5)
    g.ajouter_arc(0, 1, 4)
    g.ajouter_arc(0, 2, 1)
    g.ajouter_arc(1, 4, 4)
    g.ajouter_arc(2, 1, 1)
    g.ajouter_arc(2, 3, 5)
    g.ajouter_arc(3, 4, 3)
    return g


def creer_graphe_exemple_2() -> Graphe:
    g = Graphe(4)
    g.ajouter_arc(0, 1, 4)
    g.ajouter_arc(0, 2, 2)
    g.ajouter_arc(1, 2, 1)
    g.ajouter_arc(1, 3, -3)
    g.ajouter_arc(2, 1, -2)
    return g


def tester_algorithmes():
    print("=" * 70)
    print("TEST 1 : Dijkstra sur graphe à coûts positifs")
    print("=" * 70)
    g1 = creer_graphe_exemple_1()
    g1.afficher()
    lambda_vals, predecesseurs = dijkstra(g1, 0)
    afficher_resultats(lambda_vals, predecesseurs, 0, g1)

    print("\n" + "=" * 70)
    print("TEST 2 : Bellman-Ford sur graphe à coûts négatifs")
    print("=" * 70)
    g2 = creer_graphe_exemple_2()
    g2.afficher()
    lambda_vals, predecesseurs, circuit = bellman_ford(g2, 0)
    if circuit:
        print("\n⚠️  ATTENTION : circuit absorbant détecté !")
    else:
        afficher_resultats(lambda_vals, predecesseurs, 0, g2)

    print("\n" + "=" * 70)
    print("TEST 3 : Ford sur le même graphe")
    print("=" * 70)
    lambda_vals, predecesseurs, circuit = ford(g2, 0)
    if circuit:
        print("\n⚠️  ATTENTION : circuit absorbant détecté !")
    else:
        afficher_resultats(lambda_vals, predecesseurs, 0, g2)

    print("\n" + "=" * 70)
    print("TEST 4 : Floyd-Warshall")
    print("=" * 70)
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
                print(f"  v{i} → v{j}: distance = {distances[i,j]:.1f}, chemin: {chemin_str}")


if __name__ == "__main__":
    tester_algorithmes()
