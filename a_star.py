import math
from heapq import heappush, heappop

import numpy as np


class PriorityQueue:

    def __init__(self, iterable=[]):
        self.heap = []
        for value in iterable:
            heappush(self.heap, (0, value))

    def add(self, value, priority=0):
        heappush(self.heap, (priority, value))

    def pop(self):
        priority, value = heappop(self.heap)
        return value

    def __len__(self):
        return len(self.heap)


def get_heuristic(h_fun, dim):
    def calc_h(cell):
        (i, j) = cell
        if h_fun == 'MANHATTAN':
            return abs(dim - i) + abs(dim - j)
        elif h_fun == 'EUCLIDEAN':
            return math.sqrt(abs(dim - i) ** 2 + abs(dim - j) ** 2)
        elif h_fun == 'CHEBYSHEV':
            return max(abs(dim - i), abs(dim - j))
        else:
            return max(abs(dim - i), abs(dim - j))

    return calc_h


def a_star_search(start, end, neighbors, heuristic, grid):
    dim = len(grid[0])
    visited = set()
    parent = dict()
    distance = {start: 0}
    fringe = PriorityQueue()
    fringe.add(start)
    while fringe:
        cell = fringe.pop()
        if cell in visited:
            continue
        if cell == end:
            return reconstruct_path(parent, start, cell)
        visited.add(cell)
        for child in neighbors(cell):
            fringe.add(child, priority=distance[cell] + 1 + heuristic(child))
            if child not in distance or distance[cell] + 1 < distance[child]:
                distance[child] = distance[cell] + 1
                parent[child] = cell
    return None


def reconstruct_path(parent, start, end):
    path = [end]
    while end != start:
        end = parent[end]
        path.append(end)
    return list(reversed(path))


def get_neighbors(grid, dim):
    def get_adjacent_cells(cell):
        x, y = cell
        return ((x + i, y + j)
                for (i, j) in [(1, 0), (0, 1), (0, -1), (-1, 0)]
                # (i, j) Represents movement from current cell - N,W,S,E direction eg: (1,0) means -> (x+1, y)
                # neighbor should be within grid boundary
                # neighbor should be an unblocked cell
                if 0 <= x + i < dim
                if 0 <= y + j < dim
                if grid[x + i][y + j] == 0)

    return get_adjacent_cells


def get_shortest_path(grid, start, end):
    h_fun = 'MANHATTAN'
    # print('### START A * ###')
    grid = np.where(grid == 0, 1, grid)
    grid = np.where(grid < 1, 0, grid)
    # print('### END A * ###')
    dim = len(grid[0])
    shortest_path = a_star_search(start, end, get_neighbors(grid, dim), get_heuristic(h_fun, dim), grid)
    if shortest_path is None:
        return -1
    else:
        return shortest_path
