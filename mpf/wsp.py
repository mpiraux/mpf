# Code adapted from https://github.com/louisna/rust-wsp/ by Louis Navarre

import sys
import dataclasses
import random
from typing import List, Tuple

from scipy.spatial.distance import cdist
import numpy as np


@dataclasses.dataclass
class PointSet:
    points: np.ndarray
    distance_matrix: np.ndarray
    active: np.ndarray
    no_active: int

    idx_active: np.ndarray
    visited: np.ndarray
    d_min: float
    d_max: float

    @classmethod
    def from_preset(cls, points: List[List[float]]) -> "PointSet":
        distance_matrix, d_min, d_max = PointSet.compute_distance_matrix(points)
        ps = PointSet(
            points=np.array(points, dtype=np.double),
            distance_matrix=distance_matrix,
            active=np.array([True] * len(points), dtype=np.bool_),
            no_active=len(points),
            idx_active=np.array([1] * len(points), dtype=np.intc),
            visited=np.array([False] * len(points), dtype=np.bool_),
            d_min=d_min,
            d_max=d_max,
        )
        return ps

    @classmethod
    def from_random(cls, no_points: int, no_dim: int, seed: str) -> "PointSet":
        points = []
        rng = random.Random(seed)
        for _ in range(no_points):
            points.append([rng.random() for _ in range(no_dim)])
        return cls.from_preset(points)

    @classmethod
    def compute_distance_matrix(
        cls, points: List[List[float]], compute_distance="cityblock"
    ) -> Tuple[np.ndarray, float, float]:
        distance_matrix = cdist(points, points, compute_distance)  # type: ignore
        mask = np.ones(distance_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        dmin = distance_matrix[mask].min()
        dmax = distance_matrix[mask].max()
        return distance_matrix, dmin, dmax

    def reset_research_params(self):
        self.no_active = len(self.points)
        self.active = np.array([True] * len(self.points), dtype=np.bool_)
        self.idx_active = np.array([1] * len(self.points), dtype=np.intc)
        self.visited = np.array([False] * len(self.points), dtype=np.bool_)

    def get_closest_idx(self, i: int) -> np.ndarray:
        return np.argsort(self.distance_matrix[i])

    def get_remaining(self) -> List[List[float]]:
        return [
            self.points[i].copy() for i in range(len(self.points)) if self.active[i]
        ]

    def wsp_loop_fast(self, d_min: float, origin: int):
        while True:
            idxs_this_origin = self.get_closest_idx(origin)
            closest_origin = self.idx_active[origin]
            self.visited[origin] = True

            while True:
                if closest_origin >= len(self.points):
                    return

                point_idx = idxs_this_origin[closest_origin]
                if not self.active[point_idx]:
                    closest_origin += 1
                    continue
                elif self.distance_matrix[origin][point_idx] < d_min:
                    self.active[point_idx] = False
                    self.no_active -= 1
                    closest_origin += 1
                elif self.visited[point_idx]:
                    closest_origin += 1
                else:
                    self.idx_active[origin] = closest_origin
                    origin = idxs_this_origin[closest_origin]  # type: ignore
                    break

    def wsp(self, d_min: float):
        rng = random.Random("wsp")
        origin = rng.randint(0, len(self.points) - 1)
        self.wsp_loop_fast(d_min, origin)

    def adaptive_wsp(self, target: int, verbose=False):
        d_min = self.d_min
        d_max = self.d_max
        d_search = (d_min + d_max) / 2
        it = 0
        best_distance = 0.0
        best_difference_active = self.no_active - target

        while True:
            it += 1
            self.wsp(d_search)

            if verbose:
                print(
                    "Iter #{}: distance={}, no_active={}".format(
                        it, d_search, self.no_active
                    )
                )

            if self.no_active > target:
                d_min = d_search
            elif self.no_active < target:
                d_max = d_search
            else:
                return

            if abs(self.no_active - target) < best_difference_active:
                best_difference_active = abs(self.no_active - target)
                best_distance = d_search

            last_d_search = d_search
            d_search = (d_min + d_max) / 2
            if abs(last_d_search - d_search) < sys.float_info.epsilon:
                break

            self.reset_research_params()

        if abs(best_distance - d_search) > sys.float_info.epsilon:
            d_search = best_distance
            self.reset_research_params()
            self.wsp(d_search)

        if verbose:
            print(
                "Last iter: best approximation is distance={}, nb_active={}".format(
                    d_search, self.no_active
                )
            )


if __name__ == "__main__":
    p1 = [0.0, 0.0]
    p2 = [4.0, 0.0]
    p3 = [4.0, 3.0]
    d_mat, d_min, d_max = PointSet.compute_distance_matrix([p1, p2, p3], "sqeuclidean")

    true_dist = [[0.0, 16.0, 25.0], [16.0, 0.0, 9.0], [25.0, 9.0, 0.0]]

    for i in range(3):
        for j in range(3):
            assert d_mat[i][j] == true_dist[i][j]

    assert d_min == 9.0
    assert d_max == 25.0

    p1 = [0.0, 0.0]
    p2 = [1.0, 0.1]
    p3 = [1.0, 1.0]
    p4 = [2.0, 1.0]
    ps = PointSet.from_preset([p1, p2, p3, p4])

    true_idxs = [[0, 1, 2, 3], [1, 2, 0, 3], [2, 1, 3, 0], [3, 2, 1, 0]]

    for i in range(4):
        for j in range(4):
            assert ps.get_closest_idx(i)[j] == true_idxs[i][j]

    p1 = [0.0, 0.0]
    p2 = [1.0, 0.1]
    p3 = [1.0, 1.0]
    p4 = [2.0, 1.0]
    ps = PointSet.from_preset([p1, p2, p3, p4])

    ps.wsp_loop_fast(1.0, 1)

    assert ps.active[0]
    assert ps.active[1]
    assert not ps.active[2]
    assert ps.active[3]
    assert ps.no_active == 3

    d_min = 0.04
    ps = PointSet.from_random(1000, 3, "test")
    ps.wsp(d_min)

    for i in range(1000):
        assert ps.visited[i] or not ps.active[i]

    d_min = 0.04
    ps = PointSet.from_random(1000, 3, "test")
    ps.wsp(d_min)

    for i in range(999):
        if not ps.active[i]:
            continue
        for j in range(i + 1, 1000):
            if not ps.active[j]:
                continue
            assert ps.distance_matrix[i][j] >= d_min

    ps = PointSet.from_random(10000, 4, "test")
    ps.adaptive_wsp(100, True)

    print("All tests passed")
