import random
from typing import List, Tuple
from SingleAgentTests.Agent import OnlineAgent
from SingleAgentTests.Types import State, Action, ActionSet


class TD0(OnlineAgent):

    def __init__(self, alpha: float, epsilon: float, gamma: float, initailState: State, possibleActions: ActionSet, possibleStateActions: List[Tuple[State, Action]]):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.possibleActions = possibleActions
        self.currentState = initailState
        self.possibleStateActions = possibleStateActions
        self.q = { stateAction: 0 for stateAction in self.possibleStateActions }
        super().__init__()

    def step(self, observableState: State, possibleActions: ActionSet, reward: float) -> None:
        super().step(observableState, possibleActions, reward)
        if self.lastState == (4,4):
            print("Updating terminal state")
        self.q[(self.lastState, self.lastAction)] = self.q[(self.lastState, self.lastAction)] + self.alpha*(reward + self.gamma*self.q[(self.currentState, self.currentAction)] - self.q[(self.lastState, self.lastAction)])
        self.possibleActions = possibleActions

    def generateNextAction(self) -> None:
        actionValues = {action:self.q[(state, action)] for state, action in self.q.keys() if state == self.currentState}
        bestAction = max(actionValues, key= lambda key: actionValues[key])
        m = len(self.possibleActions)
        weights = [self.epsilon/m if i != bestAction else self.epsilon/m + 1 - self.epsilon for i in range(m)]
        self.lastAction = self.currentAction
        self.currentAction = random.choices(self.possibleActions, weights=weights)[0]
        print(f"Action values: {actionValues}, Best action: {bestAction}, Actual action: {self.currentAction}")

    def nextEpisode(self, state: State) -> None:
        super().nextEpisode(state)

        
                