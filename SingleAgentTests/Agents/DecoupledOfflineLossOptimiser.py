import random
import numpy as np
from typing import List, Tuple
from SingleAgentTests.Agent import OfflineAgent
from Common.Types import State, Action, ActionSet


class DecoupledOfflineLossOptimiser(OfflineAgent):

    def __init__(self, trainingInterval: int, epsilon: float, epsilon2: float, gamma: float, initailState: State, possibleActions: ActionSet, possibleStateActions: List[Tuple[State, Action]]):
        self.episodeNumber = 1
        self.epsilon = epsilon
        self.epsilon2 = epsilon2
        self.gamma = gamma
        self.possibleActions = possibleActions
        self.currentState = initailState
        self.possibleStateActions = possibleStateActions
        self.q = { stateAction: 0 for stateAction in self.possibleStateActions }
        self.indexStateActions = self.q.keys()
        self.stateActionIndicies = dict(zip(self.indexStateActions,[i for i in range(len(self.possibleStateActions))]))
        self.oldq = self.q.copy()
        self.oldqvec = np.zeros((len(self.indexStateActions),1 ))
        self.d = []
        self.numState0s = max([stateAction[0][0] for stateAction in self.possibleStateActions])
        self.numState1s = max([stateAction[0][1] for stateAction in self.possibleStateActions])
        self.numActions = max([stateAction[1] for stateAction in self.possibleStateActions])
        super().__init__(trainingInterval)


    def step(self, observableState: State, possibleActions: ActionSet, reward: float) -> None:
        # print(f"State: {observableState}")
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
        # print(f"Best Action: {bestAction}, Actual action: {self.currentAction}")

    def nextEpisode(self, state: State) -> None:
        super().nextEpisode(state)


    def getData(self) -> List[Tuple[State,Action,int,State,Action]]:
        numDataPoints = 400000
        # if len(self.d) > numDataPoints:
        #     return random.choices(self.d, k = numDataPoints)
        return self.d[-numDataPoints:]
    
    def indeciseStateAction(self, stateAction):
        return self.stateActionIndicies[stateAction]

    def indextToStateAction(self, index):
        return self.indexStateActions[index]

    def train(self) -> None:
        data = self.getData()
        numStateActions = len(self.possibleStateActions)
        A = self.epsilon2*np.eye(numStateActions)
        b = np.zeros((numStateActions,1))
        for d in data:
            j = self.indeciseStateAction(d[0:2])
            jprime = self.indeciseStateAction(d[3:])
            em = np.zeros((numStateActions,1))
            emprime = np.zeros((numStateActions,1))
            em[j, 0] = 1
            emprime[jprime, 0] = 1
            A += np.matmul(em,np.transpose(em))
            b += em*(d[2]+self.gamma*np.matmul(np.transpose(emprime),self.oldqvec))
        np.set_printoptions(threshold=np.inf)
        self.oldq = self.q.copy()
        qvec = np.linalg.solve(A,b)
        self.oldqvec = qvec
        print(self.oldqvec)
        self.q = dict(zip(self.indexStateActions, qvec.reshape(numStateActions)))
        # print(f"Qaftervec = {self.q}")
        # self.epsilon /= 1.05

    def calculateLoss(self, d) -> float:
        loss = 0
        for s,a,r,sprime,aprime in d:
            loss += (self.q[(s,a)] - r - self.gamma*self.q[(sprime,aprime)])**2
        loss /= len(d)
        return loss

    
