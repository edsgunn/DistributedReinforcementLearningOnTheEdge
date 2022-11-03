import random
from typing import List, Tuple
from SingleAgentTests.Agent import Agent
from SingleAgentTests.Types import State, Action, ActionSet


class TDlambda(Agent):

    def __init__(self, alpha: float, epsilon: float, gamma: float, lambd: float, initailState: State, possibleActions: ActionSet, possibleStateActions: List[Tuple[State, Action]]):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.lambd = lambd
        self.possibleActions = possibleActions
        self.currentState = initailState
        self.possibleStateActions = possibleStateActions
        self.q = { stateAction: 0 for stateAction in self.possibleStateActions }
        self.e = { stateAction: 0 for stateAction in self.possibleStateActions }
        self.currentAction = None
        self.lastAction = None
        self.generateNextAction()

    def step(self, observableState: State, possibleActions: ActionSet, reward: float) -> None:
        print(f"State: {observableState}")
        print(f"Possible actions: {possibleActions}")
        print(f"Reward: {reward}")
        self.lastState = self.currentState
        self.currentState = observableState
        self.generateNextAction()
        if self.lastAction is not None:
            delta = reward + self.gamma*self.q[(self.currentState, self.currentAction)]-self.q[(self.lastState, self.lastAction)]
            self.e[(self.lastState, self.lastAction)] += 1
            for stateAction in self.q.keys():
                self.q[stateAction] += self.alpha*delta*self.e[stateAction]
                self.e[stateAction] *= self.gamma*self.lambd
        self.possibleActions = possibleActions

    def getAction(self) -> Action:
        return self.currentAction

    def generateNextAction(self) -> None:
        actionValues = {action:self.q[(state, action)] for state, action in self.q.keys() if state == self.currentState}
        bestAction = max(actionValues, key= lambda key: actionValues[key])
        m = len(self.possibleActions)
        weights = [self.epsilon/m if i != bestAction else self.epsilon/m + 1 - self.epsilon for i in range(m)]
        self.lastAction = self.currentAction
        self.currentAction = random.choices(self.possibleActions, weights=weights)[0]
        print(f"Action values: {actionValues}, Best action: {bestAction}, Actual action: {self.currentAction}")

    def nextEpisode(self) -> None:
        self.e = { stateAction: 0 for stateAction in self.possibleStateActions }

    def getV2D(self):
        v = []
        for stateAction in self.q.keys():
            x = stateAction[0][0]
            y = stateAction[0][1]
            while len(v) - 1 < x:
                v.append([])
            while len(v[x]) - 1 < y:
                v[x].append(None)
            if v[x][y] is not None:
                v[x][y] = max(v[x][y], self.q[stateAction])  
            else:
                v[x][y] = self.q[stateAction]   
        return v

    def printV2D(self):
        text = "V:\n"
        v = self.getV2D()
        for row in v:
            for entry in row:
                text += f"{entry} "
            text += "\n"
        print(text)
        
                