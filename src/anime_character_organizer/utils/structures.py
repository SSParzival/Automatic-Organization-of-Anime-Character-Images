"""
Custom data structures: Disjoint Set (Union-Find) and Burkhard-Keller Metric Tree (BK-Tree).
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


class UnionFind:
    """Disjoint-set data structure with path compression and union by rank."""

    def __init__(self, items: Iterable[Any]):
        self.parent: Dict[Any, Any] = {item: item for item in items}
        self.rank: Dict[Any, int] = {item: 0 for item in items}

    def find(self, item: Any) -> Any:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: Any, b: Any) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return

        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]

        if rank_a < rank_b:
            self.parent[root_a] = root_b
        elif rank_a > rank_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


class BKTree:
    """
    Burkhard-Keller metric tree for efficient discrete metric search (e.g., Hamming distance).
    """

    def __init__(self, distance_fn: Callable[[Any, Any], int]):
        self.distance = distance_fn
        self.root: Optional[List[Any]] = None

    def add(self, item: Any) -> None:
        if self.root is None:
            self.root = [item, {}]
            return

        node = self.root
        while True:
            current_item = node[0]
            children = node[1]
            dist = self.distance(item, current_item)

            if dist in children:
                node = children[dist]
            else:
                children[dist] = [item, {}]
                return

    def query(self, item: Any, threshold: int) -> List[Tuple[Any, int]]:
        if self.root is None:
            return []

        results: List[Tuple[Any, int]] = []
        nodes = [self.root]

        while nodes:
            current_node, children = nodes.pop()
            dist = self.distance(item, current_node)

            if dist <= threshold:
                results.append((current_node, dist))

            low = dist - threshold
            high = dist + threshold

            for child_dist, child in children.items():
                if low <= child_dist <= high:
                    nodes.append(child)

        return results
