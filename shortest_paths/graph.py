from typing import List, Dict, Tuple
from collections import defaultdict


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
        self.n = n
        self.arcs: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    def ajouter_arc(self, i: int, j: int, cout: float):
        """Ajoute un arc (vi, vj) de coût Cij."""
        self.arcs[i].append((j, cout))

    def obtenir_successeurs(self, i: int) -> List[Tuple[int, float]]:
        """Retourne la liste des successeurs du sommet i."""
        return self.arcs[i]

    def obtenir_tous_arcs(self) -> List[Tuple[int, int, float]]:
        """Retourne tous les arcs du graphe."""
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
