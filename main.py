from datetime import datetime

import numpy as np

import a_star
import board

blocked = []
parent = dict()


class player(object):
    global blocked, parent

    def __init__(self, board, agent=6, maxIter=2000000):
        # board.board board: game board
        # bool quickRes: True: faster calculation; False: more precise result
        # int agent in [1 : 3]: search strategy.
        # int maxIter in [1 : inf]: max search times in a board
        self.b = board
        self.agent = agent
        self.maxIter = maxIter
        # double check related
        self.doubleCount = np.zeros_like(self.b.cell, dtype=np.uint8)
        # report related
        self.reportHistory = []
        self.targetHistory = []
        self.searchHistory = []

        self.success = False
        self.history = []
        return

    # update functions
    # update b.prob after search
    def updateP(self, prob, row, col, quick=False, force=False, temp=False, blocked=[]):
        # bool force: True: force to normalize prob; False: depand on quick
        # bool temp: True: this is a temp prob, and will not update b.prob; False: update b.prob
        # process temp
        if temp:
            tempProb = np.copy(prob)
        else:
            tempProb = prob

        # update
        if (row, col) in blocked:
            tempProb[row, col] = 0
        else:
            tempProb[row, col] = tempProb[row, col] * self.b.failP[self.b.cell[row, col]]

        # process quick
        sumP = np.sum(tempProb)
        if not force and sumP > 0.5:
            if not temp:
                self.b.prob = tempProb
            return tempProb
        tempProb = self.normalizeP(tempProb, sumP)
        if not temp:
            self.b.prob = tempProb
        return tempProb

    # update b.prob after report
    def updateR(self, prob, report, quick=False, force=False, temp=False):
        findFlag, targetMove = self.findReport(report)
        if findFlag:
            # print('re-update')
            tempProb = self.reUpdateReport(temp=temp)
        else:
            tempProb = self.updateReport(prob, targetMove, quick=quick, force=force, temp=temp)
        return tempProb

    # report functions
    # analysis report history
    def findReport(self, report):
        # returns:
        # bool find Flag: True: reportHistory can translate to targetHistory; False: cannot translate
        # tuple targetMove with element (prev, post): target move from prev to post
        findFlag = False
        if self.targetHistory:  # translatable
            targetMove = self.findTarget(report)
        elif self.reportHistory:
            tPrevTer = self.reportHistory[-1] * report  # try to translate
            if 1 == np.count_nonzero(tPrevTer):  # translatable
                self.backtrackReport(tPrevTer)
                targetMove = self.findTarget(report)
                findFlag = True
            elif 2 == np.count_nonzero(tPrevTer):  # not translatable
                tReport = tuple(np.where(report > 0)[0])
                if len(tReport) == 1:
                    tReport = (tReport[0], tReport[0])
                targetMove = (tReport, tReport[:: -1])
            else:
                exit()
        else:
            tReport = tuple(np.where(report > 0)[0])
            if len(tReport) == 1:
                tReport = (tReport[0], tReport[0])
            targetMove = (tReport, tReport[:: -1])

        self.reportHistory.append(report)
        return (findFlag, targetMove)

    # update temp report
    def updateReport(self, prob, targetMove, quick=False, force=False, temp=False):
        tempProb = np.zeros_like(prob, dtype=np.float16)
        # for each possible move
        for prev, post in targetMove:
            # for each possible prev block
            for row in range(self.b.rows):
                for col in range(self.b.cols):
                    if self.b.cell[row, col] == prev:
                        # update each possible post block
                        index = tuple(np.where(self.b.border[row, col, post, :])[0])
                        factor = len(index)
                        if factor:
                            nPos = ((row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col))
                            tempP = prob[row, col] / factor
                            for i in index:
                                tempProb[nPos[i]] = tempProb[nPos[i]] + tempP

        sumP = np.sum(tempProb)

        if not force and sumP > 0.5:
            if not temp:
                self.b.prob = tempProb
            return tempProb
        tempProb = self.normalizeP(tempProb, sumP)
        if not temp:
            self.b.prob = tempProb
        return tempProb

    # re-update all report
    def reUpdateReport(self, temp=False):
        history = list(map(lambda x: np.where(x)[0][0], self.targetHistory))
        tempProb = np.full((self.b.rows, self.b.cols), (1. / (self.b.rows * self.b.cols)), dtype=np.float16)
        for i in range(len(history) - 1):
            tempProb = self.updateP(tempProb, *self.searchHistory[i], quick=True, temp=temp)
            tempProb = self.updateReport(self.b.prob, ((history[i], history[i + 1]),), quick=True, temp=temp)
        if not temp:
            self.b.prob = tempProb
        return tempProb

    # tool functions
    # resize prob so that sum == 1
    def normalizeP(self, tempProb, sumP=None):
        if sumP is None:
            sumP = np.sum(tempProb)
        if sumP == 0:
            # print('E: solution.normalizeP. zero sumP')
            exit()
        tempProb = tempProb / sumP
        return tempProb

    def moveTo(self, row, col):
        a_star_path = a_star.get_shortest_path(grid=self.b.sucP, start=self.b.robot, end=(row, col))
        for x in a_star_path:
            self.history.append((x, 'm'))
        self.b.robot = (row, col)
        return

    def search(self, row, col):
        # move or teleport
        if self.b.moving:
            self.moveTo(row, col)

        # explore
        self.searchHistory.append((row, col))
        self.history.append(((row, col), 's'))
        return self.b.explore(row, col)

    # report tool functions
    # get temp target movement
    def findTarget(self, report):
        diff = (report - self.targetHistory[-1]) > 0
        tTer = np.where(diff)
        tMove = (np.where(self.targetHistory[-1])[0][0], tTer[0][0])
        self.targetHistory.append(diff)
        return (tMove,)

    # translate reportHistory to targetHistory
    def backtrackReport(self, tPrevTer):
        tTer = tPrevTer > 0
        self.targetHistory.insert(0, tTer)
        reportList = self.reportHistory.copy()
        reportList.reverse()
        for report in reportList:
            tTer = (report - tTer) > 0
            self.targetHistory.insert(0, tTer)
        return

    # get next cell
    def getNext(self, row=None, col=None, agent=3):
        pos = None
        while pos in blocked or pos is None:
            if agent == 6:
                pos = self.belAgent(row, col)
            elif agent == 7:
                pos = self.confAgent(row, col)
            elif agent == 8:
                pos = self.minMoveAgent(row, col, agent=agent)
            elif agent == 8.1:
                pos = self.minMoveAgent(row, col, agent=agent)
            elif agent == 9:
                pos = self.minCost(row, col)
            else:
                exit()
            if pos in blocked:
                pos = parent[pos]
        return pos

    # agent 6: using belief state
    def belAgent(self, row=None, col=None):
        value = self.b.prob
        temp_m_val = value.copy()
        max_p = np.unravel_index(np.argmax(temp_m_val), temp_m_val.shape)
        np.put(temp_m_val, [blocked], 0)
        while a_star.get_shortest_path(grid=self.b.sucP, start=self.b.robot, end=max_p) == -1 or max_p in blocked:
            if max_p not in blocked:
                blocked.append(max_p)
                temp_m_val[max_p] = 0
                self.b.prob[max_p] = 0
            max_p = np.unravel_index(np.argmax(temp_m_val), temp_m_val.shape)
        return max_p

    # agent 7 : using confidence state
    def confAgent(self, row=None, col=None):
        value = self.b.prob * self.b.sucP
        temp_m_val = value.copy()
        max_p = np.unravel_index(np.argmax(temp_m_val), temp_m_val.shape)
        np.put(temp_m_val, [blocked], 0)
        while a_star.get_shortest_path(grid=self.b.sucP, start=self.b.robot, end=max_p) == -1 or max_p in blocked:
            if max_p not in blocked:
                blocked.append(max_p)
                temp_m_val[max_p] = 0
                self.b.prob[max_p] = 0
            max_p = np.unravel_index(np.argmax(temp_m_val), temp_m_val.shape)
        return max_p

    # agent 8, 8.1 : minimum moves respecting the confidence state
    def minMoveAgent(self, row=None, col=None, agent=8):
        if row is None or col is None:
            row, col = self.b.robot
        find = self.b.prob * self.b.sucP
        t = np.array(self.b.dist[row, col], dtype=np.float128)
        if agent == 8:
            d = t ** 2
        elif agent == 8.1:
            d = np.exp2(t)
        else:
            # default
            d = t ** 2
        d[d == np.inf] = 0
        value = np.divide(find, d, out=np.zeros_like(find), where=d != 0)
        temp_m_val = value.copy()
        max_p = np.unravel_index(np.argmax(temp_m_val), temp_m_val.shape)
        np.put(temp_m_val, [blocked], 0)
        while a_star.get_shortest_path(grid=self.b.sucP, start=self.b.robot, end=max_p) == -1 or max_p in blocked:
            if max_p not in blocked:
                blocked.append(max_p)
                temp_m_val[max_p] = 0
                self.b.prob[max_p] = 0
            max_p = np.unravel_index(np.argmax(temp_m_val), temp_m_val.shape)
        return max_p

    # Agent 9 moving agent for moving = True, targetMoving = True #WARN: DO NOT use in targetMoving = False, because factor will be truncated
    def minCost(self, row=None, col=None):
        if row is None or col is None:
            row, col = self.b.robot
        factor = np.empty_like(self.b.prob, dtype=np.float16)
        searchCost = np.empty_like(self.b.prob, dtype=np.float16)
        movingCost = np.zeros_like(self.b.prob, dtype=np.float16)
        # base = np.sum(self.b.prob / self.b.sucP)
        base = np.sum(np.divide(self.b.prob, self.b.sucP, out=np.zeros_like(self.b.prob), where=self.b.sucP != 0))
        valid = (self.b.prob != 0)
        if self.b.targetMoving:
            factor[valid] = 1 / (1 - self.b.prob[valid] * self.b.sucP[valid])
            searchCost[valid] = (factor[valid] - 1) * (base - self.b.prob[valid])
        else:
            factor = 1 / (1 - self.b.prob * self.b.sucP)
            searchCost = (factor - 1) * (base - self.b.prob)
        if self.b.moving:
            for nRow in range(self.b.rows):
                for nCol in range(self.b.cols):
                    if valid[nRow, nCol]:
                        movingCost[nRow, nCol] = (factor[nRow, nCol] - 1) * np.sum(self.b.prob * self.b.dist[nRow, nCol])
        value = searchCost - movingCost - self.b.dist[row, col]
        value[~valid] = -np.inf
        temp_m_val = value.copy()
        max_p = np.unravel_index(np.argmax(temp_m_val), temp_m_val.shape)
        np.put(temp_m_val, [blocked], -np.inf)
        while a_star.get_shortest_path(grid=self.b.sucP, start=self.b.robot, end=max_p) == -1 or max_p in blocked:
            if max_p not in blocked:
                blocked.append(max_p)
                temp_m_val[max_p] = -np.inf
                self.b.prob[max_p] = 0
            max_p = np.unravel_index(np.argmax(temp_m_val), temp_m_val.shape)
        return max_p

    def find(self):
        pos = self.b.robot
        prev_pos = self.b.robot
        while not self.success:
            if pos in blocked:
                pos = parent[pos]
            else:
                pos = self.getNext(*self.b.robot, agent=self.agent)
                parent[pos] = prev_pos
            if self.b.sucP[pos] == 0:
                blocked.append(pos)
            else:
                prev_pos = pos
            # explore
            self.success, report = self.search(*pos)
            self.doubleCount[pos] = self.doubleCount[pos] + 1
            if self.success:
                break

            # update
            self.updateP(self.b.prob, *pos, blocked=blocked)
            if self.b.targetMoving:
                self.updateR(self.b.prob, report)
            self.b.probHistory.append(self.b.prob.copy())
            if len(self.searchHistory) > self.maxIter:
                break
        return


if __name__ == '__main__':
    for size in [5, 10, 15, 20, 25, 50, 101]:
        # f = open('data_' + str(size) + '.csv', 'w+')
        # f.write('iteration,agent,time,examinations,moves\n')
        # f.close()
        for targetTerrain in ['Flat', 'Hilly', 'Forest']:
            for it in range(1, 101):
                b = board.grid(size=size, targetTerrain=targetTerrain, moving=True, targetMoving=False)
                robot_location = b.robot
                target_location = b._target
                prob_init = b.prob
                terrainP_init = b.terrainP
                targetHistory_init = b.targetHistory
                probHistory_init = b.probHistory
                cell_init = b.cell
                sucP_init = b.sucP
                border_init = b.border
                dist_init = b.dist
                print('iteration:', it)
                for agent in [6, 7, 8, 8.1]:
                    targetMoving = False
                    moving = True
                    if agent == 9:
                        targetMoving = True
                        moving = True
                    b = board.grid(size=size, moving=moving, targetMoving=targetMoving, robot_location=robot_location,
                                   target_location=target_location,
                                   prob_init=prob_init, targetTerrain=targetTerrain, terrainP_init=terrainP_init,
                                   targetHistory_init=targetHistory_init, probHistory_init=probHistory_init, cell_init=cell_init,
                                   sucP_init=sucP_init, border_init=border_init, dist_init=dist_init)
                    f = open('mov_' + str(size) + '.csv', 'a+')
                    blocked = []
                    parent = dict()
                    print('agent:', agent)
                    p = player(b, agent=agent)
                    start = datetime.now()
                    p.find()
                    end = datetime.now()
                    difference = (end - start)
                    time = difference.total_seconds()
                    m = [(x, y) for (x, y) in p.history if y == 'm']
                    s = [(x, y) for (x, y) in p.history if y == 's']
                    if len(p.history) < 2000001:
                        f.write(str(it) + ',' + targetTerrain + ',' + str(agent) + ',' + str(time) + ',' + str(len(s)) + ',' + str(len(m)) + '\n')
                    f.close()
