from DistributedTests.CentralLearner import CentralLearner
from typing import List, Tuple
from Common.Types import State, Action

class VanillaDRLearner(CentralLearner):

    def __init__(self, alpha: float, gamma: float) -> None:
        self.experience = []
        self.alpha = alpha
        self.gamma = gamma
        self.possibleStateActions = None
        self.q = self.q = None
        super().__init__()

    def step(self):
        updates = {}
        for data in self.experience:
            currentState = data[3]
            actionValues = {action:self.q[(state, action)] for state, action in self.q.keys() if state == currentState}
            bestAction = max(actionValues, key= lambda key: actionValues[key])
            updates[data[:2]] = (updates.get(data[:2],(0,0))[0]+1, updates.get(data[:2],(0,0))[1] + data[2] + self.gamma*self.q[(data[3],bestAction)] - self.q[data[:2]])
        for update in updates.keys():
            self.q[update] += self.alpha*(1/updates[update][0])*updates[update][1]

        self.experience = []
        self.sendMessage(self.q)

    def initialize(self, possibleStateActions: List[Tuple[State, Action]]):
        self.possibleStateActions = possibleStateActions
        self.q = self.q = { stateAction: 0 for stateAction in self.possibleStateActions }

    def recieveMessage(self, message):
        self.experience.append(message)

    def getQ(self):
        return self.q

    