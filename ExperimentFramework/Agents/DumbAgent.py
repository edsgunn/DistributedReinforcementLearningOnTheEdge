from DistributedTests.Agent import Agent
from Common.Types import State, Action, ActionSet
from typing import List, Tuple
import random

class DumbAgent(Agent):

    def __init__(self, epsilon: float, initailState: State, possibleActions: ActionSet, possibleStateActions: List[Tuple[State, Action]]) -> None:
        self.epsilon = epsilon
        self.possibleActions = possibleActions
        self.currentState = initailState
        self.possibleStateActions = possibleStateActions
        self.q = { stateAction: 0 for stateAction in self.possibleStateActions }
        super().__init__()

    def step(self, observableState: State, possibleActions: ActionSet, reward: float) -> Action:
        # print(f"State: {observableState}")
        # print(f"Possible actions: {possibleActions}")
        # print(f"Reward: {reward}")
        self.lastState = self.currentState
        self.currentState = observableState
        self.generateNextAction()
        self.sendMessage((self.lastState, self.lastAction, reward, self.currentState, self.currentAction))

    def sendMessage(self, message):
        self.centralLearner.recieveMessage(message)

    def recieveMessage(self, message):
        self.q = message

    def nextEpisode(self, initialState, possibleActions) -> None:
        self.currentState = initialState
        self.possibleActions = possibleActions
        self.lastState = None
        self.lastAction = None
    
    def getAction(self) -> Action:
        return self.currentAction

    def generateNextAction(self) -> None:
        actionValues = {action:self.q[(state, action)] for state, action in self.q.keys() if state == self.currentState}
        bestAction = max(actionValues, key= lambda key: actionValues[key])
        m = len(self.possibleActions)
        weights = [self.epsilon/m if i != bestAction else self.epsilon/m + 1 - self.epsilon for i in range(m)]
        self.lastAction = self.currentAction
        self.currentAction = random.choices(self.possibleActions, weights=weights)[0]
        # print(f"Action values: {actionValues}, Best action: {bestAction}, Actual action: {self.currentAction}")



