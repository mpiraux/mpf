# Code adapted from https://github.com/louisna/rust-wsp/ by Louis Navarre

import sys
import dataclasses
import random
from typing import List, Tuple

def manhattan_distance(p1: List[float], p2: List[float]) -> float:
    return sum((abs(v1 - v2) for v1, v2 in zip(p1, p2)))

def dist(p1: List[float], p2: List[float]) -> float:
    return sum(((v1 - v2) * (v1 - v2) for v1, v2 in zip(p1, p2)))

@dataclasses.dataclass
class PointSet:
    points: List[List[float]]
    distance_matrix: List[List[float]]
    active: List[bool]
    no_active: int

    idx_sort: List[List[int]]
    idx_active: List[int]
    visited: List[bool]
    d_min: float
    d_max: float

    @classmethod
    def from_preset(cls, points: List[List[float]]) -> 'PointSet':
        distance_matrix, d_min, d_max = PointSet.compute_distance_matrix(points)
        ps = PointSet(
            points=points, 
            distance_matrix=distance_matrix, 
            active=[True] * len(points),
            no_active=len(points),
            idx_sort=[],
            idx_active=[1] * len(points),
            visited=[False] * len(points),
            d_min=d_min,
            d_max=d_max
        )
        ps.compute_closest_idx()
        return ps
    
    @classmethod
    def from_random(cls, no_points: int, no_dim: int, seed: str) -> 'PointSet':
        points = []
        rng = random.Random(seed)
        for _ in range(no_points):
            points.append([rng.random() for _ in range(no_dim)])

        return cls.from_preset(points)

    @classmethod
    def compute_distance_matrix(cls, points: List[List[float]], compute_distance=manhattan_distance) -> Tuple[List[List[float]], float, float]:
        no_points = len(points)
        distance_matrix = [[0.0] * no_points for _ in range(no_points)]
        dmin = float('inf')
        dmax = 0.0
        for i in range(no_points):
            for j in range(i + 1, no_points):
                distance_matrix[i][j] = compute_distance(points[i], points[j])
                distance_matrix[j][i] = distance_matrix[i][j]
                dmin = min(dmin, distance_matrix[i][j])
                dmax = max(dmax, distance_matrix[i][j])

        return distance_matrix, dmin, dmax
    
    def reset_research_params(self):
        self.no_active = len(self.points)
        self.active = [True] * self.no_active
        self.idx_active = [1] * self.no_active
        self.visited = [False] * self.no_active

    def compute_closest_idx(self):
       for i in range(self.no_active):
            idxs = sorted(range(self.no_active), key=lambda a: self.distance_matrix[i][a])
            self.idx_sort.append(idxs)

    def get_remaining(self) -> List[List[float]]:
        return [self.points[i].copy() for i in range(len(self.points)) if self.active[i]]
    
    def wsp_loop_fast(self, d_min: float, origin: int):
        while True:
            idxs_this_origin = self.idx_sort[origin]
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
                    origin = idxs_this_origin[closest_origin]
                    break

    def wsp(self, d_min: float):
        rng = random.Random('wsp')
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
                print("Iter #{}: distance={}, no_active={}".format(it, d_search, self.no_active))

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
            print("Last iter: best approximation is distance={}, nb_active={}".format(d_search, self.no_active))

if __name__ == "__main__":
    p1 = [1.0, 0.0]
    p2 = [0.0, 0.0]
    assert dist(p1, p2) == 1.0

    p1 = [2.0, 2.0]
    p2 = [2.0, 9.0]
    assert dist(p1, p2) == 49.0

    p1 = [0.0, 0.0, 0.0]
    p2 = [0.5, 0.5, 1.0]
    p3 = [1.0, 0.0, 0.5]
    assert manhattan_distance(p1, p2) == 2.0
    assert manhattan_distance(p1, p3) == 1.5
    assert manhattan_distance(p2, p3) == 1.5
    assert manhattan_distance(p1, p1) == 0.0

    p1 = [0.0, 0.0]
    p2 = [4.0, 0.0]
    p3 = [4.0, 3.0]
    d_mat, d_min, d_max = PointSet.compute_distance_matrix([p1, p2, p3], dist)

    true_dist = [
        [0.0, 16.0, 25.0],
        [16.0, 0.0, 9.0],
        [25.0, 9.0, 0.0]
    ]
    
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

    true_idxs = [
        [0, 1, 2, 3],
        [1, 2, 0, 3],
        [2, 1, 3, 0],
        [3, 2, 1, 0]
    ]

    for i in range(4):
        for j in range(4):
            assert ps.idx_sort[i][j] == true_idxs[i][j]
    
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
    ps = PointSet.from_random(1000, 3, 'test')
    ps.wsp(d_min)

    for i in range(1000):
        assert ps.visited[i] or not ps.active[i]
    
    d_min = 0.04
    ps = PointSet.from_random(1000, 3, 'test')
    ps.wsp(d_min)

    for i in range(999):
        if not ps.active[i]:
            continue
        for j in range(i + 1, 1000):
            if not ps.active[j]:
                continue
            assert ps.distance_matrix[i][j] >= d_min

    print("All tests passed")