from typing import Callable, TypeVar, Generic, TypeAlias
import numpy as np
import bisect
from threading import Lock
from time import time


def int_distance(x: int, y: int) -> int:
    return (x ^ y).bit_count()


def identity(x):
    return x


Query = TypeVar("Query")
Vector = TypeVar("Vector")
Index: TypeAlias = int
Layer: TypeAlias = dict[Index, "FurthestQueue"]


class HNSW(Generic[Query, Vector]):
    def __init__(
        self,
        M: int = 128,
        efConstruction: int = 128,
        m_L: float = 0.3,
        distance_func: Callable[[Query, Vector], float] = int_distance,
        query_to_vector_func: Callable[[Query], Vector] = identity,
    ):
        # Params.
        self.M = int(M)
        self.Mmax = self.M
        self.Mmax0 = self.M
        self.efConstruction = int(efConstruction)
        self.m_L = m_L

        # User Functions.
        self._distance_func = distance_func
        self._query_to_vector_func = query_to_vector_func

        # State.
        self.vectors: list[Vector] = []
        self.entry_point: list[Index] = []
        # Layer format: [ { node: [(distance(node, neighbor), neighbor)] } ]
        self.layers: list[Layer] = []

        # Tracing.
        self.search_log = {}
        self.record_search_log = False

        # Stats.
        # Precompute the count of comparison of a list bisection.
        self.n_cmp_per_len = [
            int(np.ceil(np.log2(list_length + 1))) for list_length in range(10_000)
        ]
        self._reset_stats()

    # --- State Operations ---

    def _insert_vector(self, vec: Vector) -> Index:
        with Lock():
            q = len(self.vectors)
            self.vectors.append(vec)
            self.n_insertions += 1
        return q

    def get_layer(self, lc: int) -> Layer:
        return self.layers[lc] if lc < len(self.layers) else {}

    def _mut_layer(self, lc: int) -> Layer:
        while lc >= len(self.layers):
            self.layers.append({})
        return self.layers[lc]

    @staticmethod
    def get_links(e: Index, layer) -> "FurthestQueue":
        return layer.get(e) or FurthestQueue()

    @staticmethod
    def mut_links(e, layer) -> "FurthestQueue":
        conn = layer.get(e)
        if conn is None:
            conn = FurthestQueue()
            layer[e] = conn
        return conn

    def db_size(self) -> int:
        return len(self.vectors)

    def _connect_bidir(
        self,
        q: Index,
        neighbors: list[tuple[float, Index]],
        lc: int,
        max_links: int,
    ):
        with Lock():
            layer = self._mut_layer(lc)

            # Connect q -> n.
            if q in layer:
                print("Warning: _connect_bidir: q is already in the layer.")
                return

            layer[q] = FurthestQueue(neighbors, is_ascending=True)

            for nq, n in neighbors:
                # Connect n -> q.
                n_links = HNSW.mut_links(n, layer)
                self._record_list_comparison(len(n_links))
                n_links.add(nq, q)
                if len(n_links) > max_links:
                    n_links.trim_to_k_nearest(max_links)
                    # or select_neighbors_heuristic.

    # --- Stats ---

    def _reset_stats(self):
        self.n_insertions = 0
        self.n_searches = 0
        self.n_distances = 0
        self.n_comparisons = 0
        self.n_improve = 0
        self.stat_time = time()

    def get_params(self) -> dict[str, int | float]:
        return {
            "M": self.M,
            "efConstruction": self.efConstruction,
            "m_L": self.m_L,
            # "Mmax": self.Mmax,
            # "Mmax0": self.Mmax0,
        }

    def get_stats(self) -> dict[str, int | float]:
        return {
            "db_size": len(self.vectors),
            "n_layers": len(self.layers),
            "n_insertions": self.n_insertions,
            "n_searches": self.n_searches,
            "n_distances": self.n_distances,
            "n_comparisons": self.n_comparisons,
            "n_improve": self.n_improve,
            "duration_sec": time() - self.stat_time,
        }

    def reset_stats(self) -> dict[str, int | float]:
        "Return the current stats, then reset them."
        stats = self.get_stats()
        self._reset_stats()
        return stats

    def _record_list_comparison(self, list_length: int):
        self.n_comparisons += self.n_cmp_per_len[list_length]

    # --- HNSW Algorithms ---

    def _distance(self, x_vec: Query, y_id: Index) -> float:
        self.n_distances += 1
        y_vec = self.vectors[y_id]
        return self._distance_func(x_vec, y_vec)

    def _select_layer(self) -> int:
        return int(-np.log(np.random.random()) * self.m_L)

    def insert(self, q_vec: Query) -> Index:
        q_vec_to_store = self._query_to_vector_func(q_vec)
        q = self._insert_vector(q_vec_to_store)

        W = self._search_init(q_vec)
        L = len(self.layers) - 1
        l = self._select_layer()

        # From the top layer down to the new node layer, non-inclusive.
        for lc in range(L, l, -1):
            self._search_layer(q_vec, W, 1, lc)
            W.trim_to_k_nearest(1)

        for lc in range(min(L, l), -1, -1):
            self._search_layer(q_vec, W, self.efConstruction, lc)
            neighbors = W.get_k_nearest(self.M)  # or select_neighbors_heuristic

            max_conn = self.Mmax if lc else self.Mmax0
            self._connect_bidir(q, neighbors, lc, max_conn)

        if l > L:
            while l > len(self.layers) - 1:
                self.layers.append({})
            self.entry_point[:] = [q]

        return q  # ID of the inserted vector.

    def search(
        self, q_vec: Query, K: int, ef: int | None = None
    ) -> list[tuple[float, int]]:
        "Return the K nearest neighbors of q_vec, as [(distance, vector_id)]."
        self.n_searches += 1
        if not ef:
            ef = self.efConstruction

        if K > ef:
            ef = K

        W = self._search_init(q_vec)
        L = len(self.layers) - 1

        for lc in range(L, 0, -1):
            self._search_layer(q_vec, W, 1, lc)
            W.trim_to_k_nearest(1)

        self._search_layer(q_vec, W, ef, 0)
        W.trim_to_k_nearest(K)
        return W

    def _search_init(self, q_vec: Query) -> "FurthestQueue":
        if self.record_search_log:
            self.search_log.clear()

        W = FurthestQueue()

        for e in self.entry_point:
            eq = self._distance(q_vec, e)

            self._record_list_comparison(len(W))
            W.add(eq, e)

            if self.record_search_log:
                self.search_log[e] = (len(self.search_log), 0, eq, eq)

        return W

    def _search_layer(self, q_vec: Query, W: "FurthestQueue", ef: int, lc: int):
        "Mutate W into the ef nearest neighbors of q_vec in the given layer."

        layer = self.get_layer(lc)
        v = set(e for eq, e in W)  # set of visited elements
        C = NearestQueue.from_furthest_queue(W)  # set of candidates
        fq, _ = W[-1]  # W.get_furthest()

        while len(C) > 0:
            cq, c = C.pop()  # C.take_nearest()

            if self.record_search_log:
                depth = self.search_log[c][1] + 1
            else:
                depth = None

            if cq > fq:
                break  # all elements in W are evaluated

            # update C and W
            for _ec, e in self.get_links(c, layer):
                # TODO: batch distance.
                if e not in v:
                    v.add(e)

                    eq = self._distance(q_vec, e)

                    if self.record_search_log and e not in self.search_log:
                        self.search_log[e] = (len(self.search_log), depth, eq, fq)

                    if len(W) == ef:  # W is full
                        self.n_comparisons += 1  # record_list_comparison(1)
                        if eq < fq:
                            W.pop()  # W.take_furthest()
                        else:
                            continue

                    self.n_improve += 1
                    self._record_list_comparison(len(C))
                    self._record_list_comparison(len(W))
                    C.add(eq, e)
                    W.add(eq, e)

                    fq, _ = W[-1]  # W.get_furthest()


# Sorted list.


class FurthestQueue(list[tuple[float, Index]]):
    "A list sorted in ascending order, for fast pop of the furthest element."

    def __init__(self, iterable=None, is_ascending=False):
        if not iterable:
            super().__init__()
        else:
            if not is_ascending:
                iterable = sorted(iterable)
            super().__init__(iterable)

    def add(self, dist: float, to: Index):
        bisect.insort(self, (dist, to))

    def get_furthest(self) -> tuple[float, Index]:
        return self[-1]

    def take_furthest(self) -> tuple[float, Index]:
        return self.pop()

    def get_k_nearest(self, k) -> list[tuple[float, Index]]:
        return self[:k]

    def trim_to_k_nearest(self, k):
        del self[k:]


class NearestQueue(list[tuple[float, Index]]):
    "A list sorted in descending order, for fast pop of the nearest element."

    def __init__(self, iterable=None, is_descending=False):
        if not iterable:
            super().__init__()
        else:
            if not is_descending:
                iterable = sorted(iterable, reverse=True)
            super().__init__(iterable)

    @staticmethod
    def from_furthest_queue(furthest_queue: FurthestQueue) -> "NearestQueue":
        return NearestQueue(reversed(furthest_queue), is_descending=True)

    def add(self, dist: float, to: Index):
        bisect.insort(self, (dist, to), key=lambda x: -x[0])

    def get_nearest(self) -> tuple[float, Index]:
        return self[-1]

    def take_nearest(self) -> tuple[float, Index]:
        return self.pop()
