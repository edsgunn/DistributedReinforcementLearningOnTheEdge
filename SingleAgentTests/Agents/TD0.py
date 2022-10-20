import random
from typing import List, Tuple
from SingleAgentTests.Agent import Agent
from SingleAgentTests.Types import State, Action, ActionSet


class TD0(Agent):

    def __init__(self, alpha: float, epsilon: float, gamma: float, initailState: State, possibleStateActions: List[Tuple(State, Action)]):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.currentState = initailState
        self.possibleActions: ActionSet = []
        self.possibleStateActions = possibleStateActions
        self.q = { stateAction: 0 for stateAction in self.possibleStateActions }

    def step(self, observableState: State, possibleActions: ActionSet, reward: float) -> None:
        self.lastState = self.currentState
        self.currentState = observableState
        self.generateNextAction()
        self.q[(self.currentState, self.lastAction)] = self.q[(self.lastState, self.lastAction)] + self.alpha*(reward + self.gamma*self.q[(self.currentState, self.currentAction)] - self.q[(self.lastState, self.lastAction)])
        print(f"Possible actions: {possibleActions}")
        self.possibleActions = possibleActions
        print(f"Reward: {reward}")

    def getAction(self) -> Action:
        return self.currentAction

    def generateNextAction(self) -> Action:
        actionValues = {action:self.q[(state, action)] for state, action in self.q.keys() if state == self.currentState and action in self.possibleActions}
        bestAction = max(actionValues, key= lambda key: actionValues[key])
        m = len(self.possibleActions)
        weights = [self.epsilon/m if i != bestAction else self.epsilon/m +1 - self.epsilon for i in range(m)]
        self.lastAction = self.currentAction
        self.currentAction = random.choice(self.possibleActions, weights=weights)
