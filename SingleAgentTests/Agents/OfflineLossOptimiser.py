import random
from typing import List, Tuple
from SingleAgentTests.Agent import OfflineAgent
from SingleAgentTests.Types import State, Action, ActionSet


class OfflineLossOptimiser(OfflineAgent):

    def __init__(self, trainingInterval: int, epsilon: float, gamma: float, initailState: State, possibleActions: ActionSet, possibleStateActions: List[Tuple[State, Action]]):
        self.trainingInterval = trainingInterval
        self.episodeNumber = 1
        self.epsilon = epsilon
        self.gamma = gamma
        self.possibleActions = possibleActions
        self.currentState = initailState
        self.possibleStateActions = possibleStateActions
        self.q = { stateAction: 0 for stateAction in self.possibleStateActions }
        self.oldq = self.q.copy()
        self.d = []

    def step(self, observableState: State, possibleActions: ActionSet, reward: float) -> None:
        self.lastState = self.currentState
        self.currentState = observableState
        self.generateNextAction()
        self.possibleActions = possibleActions
        self.d.append((self.lastState, self.lastAction, reward, self.currentState, self.currentAction))
        

    def generateNextAction(self) -> None:
        actionValues = {action:self.oldq[(state, action)] for state, action in self.oldq.keys() if state == self.currentState}
        bestAction = max(actionValues, key= lambda key: actionValues[key])
        m = len(self.possibleActions)
        weights = [self.epsilon/m if i != bestAction else self.epsilon/m + 1 - self.epsilon for i in range(m)]
        self.lastAction = self.currentAction
        self.currentAction = random.choices(self.possibleActions, weights=weights)[0]
        print(f"Action values: {actionValues}, Best action: {bestAction}, Actual action: {self.currentAction}")

    def nextEpisode(self, state: State) -> None:
        super().nextEpisode(state)

    def train(self):
        pass

    
