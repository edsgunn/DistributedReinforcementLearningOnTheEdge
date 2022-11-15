import random
import numpy as np
from typing import List, Tuple
from SingleAgentTests.Agent import OfflineAgent
from SingleAgentTests.Types import State, Action, ActionSet


class OfflineLossOptimiser(OfflineAgent):

    def __init__(self, trainingInterval: int, epsilon: float, gamma: float, initailState: State, possibleActions: ActionSet, possibleStateActions: List[Tuple[State, Action]]):
        self.episodeNumber = 1
        self.epsilon = epsilon
        self.gamma = gamma
        self.possibleActions = possibleActions
        self.currentState = initailState
        self.possibleStateActions = possibleStateActions
        self.q = { stateAction: random.random() for stateAction in self.possibleStateActions }
        self.oldq = self.q.copy()
        self.d = []
        self.numState0s = max([stateAction[0][0] for stateAction in self.possibleStateActions])
        self.numState1s = max([stateAction[0][1] for stateAction in self.possibleStateActions])
        self.numActions = max([stateAction[1] for stateAction in self.possibleStateActions])
        super().__init__(trainingInterval)


    def step(self, observableState: State, possibleActions: ActionSet, reward: float) -> None:
        print(f"State: {observableState}")
        print(f"Possible actions: {possibleActions}")
        print(f"Reward: {reward}")
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


    def getData(self) -> List[Tuple[State,Action,int,State,Action]]:
        return self.d[54:90]
    
    def indeciseStateAction(self, stateAction):
        
        state = stateAction[0]
        action = stateAction[1]
        return state[0]*self.numState1s*self.numActions + state[1]*self.numActions + action

    def indextToStateAction(self, index):
        a = index % self.numActions
        s1 = (index-a)/self.numActions % self.numState1s
        s0 = ((index-a)/self.numActions - s1)/self.numState1s
        return ((s0,s1),a)

    def train(self) -> None:
        data = self.getData()
        numDataPoints = len(data)
        numStateActions = len(self.possibleStateActions)
        Em = np.zeros((numDataPoints,numStateActions))
        Emprime = np.zeros((numDataPoints,numStateActions))
        R = np.zeros(numDataPoints)
        for i, d in enumerate(data):
            j = self.indeciseStateAction(d[0:2])
            jprime = self.indeciseStateAction(d[3:])
            Em[i, j] += 1
            Emprime[i, jprime] += 1
            R[i]+= d[2]
        E = Em-self.gamma*Emprime
        A = 2*np.matmul(E,np.transpose(E))
        A = A + 0.01*np.eye(np.size(A,0))
        np.set_printoptions(threshold=np.inf)
        print(f"A matrix: {A} \n Size: {np.shape(A)}")
        self.oldq = self.q.copy()
        qvec = np.linalg.solve(A,R)
        for index, value in enumerate(qvec):
            self.q[self.indextToStateAction(index)] = value
        self.epsilon /= 1.5

    def calculateLoss(self, d) -> float:
        loss = 0
        for s,a,r,sprime,aprime in d:
            loss += (self.q[(s,a)] - r - self.gamma*self.q[(sprime,aprime)])**2
        loss /= len(d)
        return loss

    
