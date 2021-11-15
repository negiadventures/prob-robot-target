import numpy as np

import a_star


class grid(object):
    # 0.7/3 for each terrain, 0.3 for block
    def __init__(self, size=50, terrainP=[0.24, 0.23, 0.23, 0.3], failP=[0.2, 0.5, 0.8, 1], moving=False, targetMoving=False, robot_location=None,
                 target_location=None, prob_init=None, terrainP_init=None,
                 targetHistory_init=None, probHistory_init=None, cell_init=None,
                 sucP_init=None, border_init=None, dist_init=None):
        # int size in [2 : inf]: size of board
        # list terrainP with element float p: P(terrain)
        # list failP with element float p: P(false negative | terrain)
        # bool moving: True: robot have to move between blocks; False: robot can teleport
        # bool targetMoving: True: target move every failed search; False: stationary target.
        self.rows = size
        self.cols = size
        self.moving = moving
        self.targetMoving = targetMoving
        self.failP = failP
        self.search = True
        # initialize probabilities
        if prob_init is None:
            self.border = np.zeros((self.rows, self.cols, 4, 4), dtype=np.bool)
            self.dist = np.zeros((self.rows, self.cols, self.rows, self.cols), dtype=np.uint8)
            self.prob = np.full((self.rows, self.cols), (1. / (self.rows * self.cols)), dtype=np.float16)
            self.cell = np.empty((self.rows, self.cols), dtype=np.uint8)
            self.sucP = np.empty((self.rows, self.cols), dtype=np.float16)
            self.terrainP = []
            self.getTerrainP(terrainP)
            self.buildTerrain()
            self.targetHistory = []
            self.probHistory = []
            self.probHistory.append(self.prob.copy())
            self.getDist()
            self.robot = (self.rows // 2, self.cols // 2)
            self.robotPos()
            self._target = (self.rows, self.cols)
            self.hideTarget()
            while a_star.get_shortest_path(grid=self.sucP, start=self.robot, end=self._target) == -1:
                self.robotPos()
                self.hideTarget()
        else:
            self.prob = prob_init
            self.cell = cell_init
            self.sucP = sucP_init
            self.terrainP = terrainP_init
            self.targetHistory = targetHistory_init
            self.probHistory = probHistory_init
            self.robot = robot_location
            self._target = target_location
            self.border = border_init
            self.dist = dist_init
        return

    def getTerrainP(self, terrainP):
        sumP = 0
        for p in terrainP:
            self.terrainP.append(sumP)
            sumP = sumP + p
        return

    # init cell
    def buildTerrain(self):
        terrain = np.random.rand(self.rows, self.cols)
        for i in range(len(self.terrainP)):
            self.cell[terrain >= self.terrainP[i]] = i
            self.sucP[terrain >= self.terrainP[i]] = 1 - self.failP[i]
        if self.targetMoving:
            self.getBorder()
        return

    def robotPos(self):
        pos = int(np.floor(np.random.random() * self.rows * self.cols))
        row, col = divmod(pos, self.cols)
        # TODO: Position of target should not fall on blocked cell
        while self.sucP[row][col] == 0:
            pos = int(np.floor(np.random.random() * self.rows * self.cols))
            row, col = divmod(pos, self.cols)
        self.robot = (row, col)
        return

    # init _target
    def hideTarget(self):
        pos = int(np.floor(np.random.random() * self.rows * self.cols))
        row, col = divmod(pos, self.cols)
        # Position of target should not fall on blocked cell
        while self.sucP[row][col] == 0:
            pos = int(np.floor(np.random.random() * self.rows * self.cols))
            row, col = divmod(pos, self.cols)
        # There should be a path from target to robot
        self._target = (row, col)
        self.targetHistory.append(self._target)
        return

    # init border
    def getBorder(self):
        for row in range(self.rows):
            for col in range(self.cols):
                index = 0
                for nRow, nCol in ((row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col)):
                    if 0 <= nRow < self.rows and 0 <= nCol < self.cols:
                        if self.sucP[nRow, nCol] != 0:
                            self.border[row, col, :, index][self.cell[nRow, nCol]] = True
                    index = index + 1
        return

    # init dist
    def getDist(self):
        for row in range(self.rows):
            for col in range(self.cols):
                self.dist[row, col] = self.getBlockDist(row, col)
        return

    # generate neighbor of (row, col)
    def getNeighbor(self, row, col):
        # int row in [0 : rows-1]: position x
        # int col in [0 : cols-1]: position y
        # return:
        # list neighbor with element ((row, col), index): this block's neighbor
        neighbor = []
        for (i, j) in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
            if 0 <= row + i < len(self.sucP):
                if 0 <= col + j < len(self.sucP):
                    if self.sucP[row + i, col + j] > 0:
                        cell = (row + i, col + j)
                        neighbor.append(cell)
        return neighbor

    def manhattan(self, x, y):
        return abs(x[0] - y[0]) + abs(x[1] - y[1])

    def getBlockDist(self, row=None, col=None):
        if row is None or col is None:
            row, col = self.robot
        dist = np.empty_like(self.cell, dtype=np.uint8)
        for tRol in range(self.rows):
            for tCol in range(self.cols):
                dist[tRol, tCol] = self.manhattan(x=(row, col), y=(tRol, tCol))
        return dist

    # action functions
    # search (row, col)
    def explore(self, row=None, col=None):
        # int row: position
        # int col: position
        # return:
        # bool res: True: done! False: not found
        # np.ndarray report with shape = (4, ): target moving report

        # teleport
        if row is None or col is None:
            pass
        else:
            self.robot = (row, col)
        self.search = True

        # search
        if self.robot == self._target:
            if np.random.random() < self.failP[self.cell[self.robot]]:
                report = self.targetMove()
                return (False, report)
            else:
                return (True, None)
        else:
            report = self.targetMove()
            return (False, report)

    # move to (row, col)
    def move(self, row, col):
        # int row: position
        # int col: position

        self.search = False
        self.robot = (row, col)
        return

    # target move to neighbor
    def targetMove(self):
        # return:
        # np.ndarray report with shape = (4, ): target moving report
        report = np.zeros((4,), dtype=np.uint8)
        if self.targetMoving:
            candidate = self.getNeighbor(*self._target)
            index = int(np.floor(np.random.random() * len(candidate)))
            report[self.cell[self._target]] = report[self.cell[self._target]] + 1
            report[self.cell[candidate[index]]] = report[self.cell[candidate[index]]] + 1
            self._target = candidate[index]
            self.targetHistory.append(self._target)
        # print(report)
        return report
