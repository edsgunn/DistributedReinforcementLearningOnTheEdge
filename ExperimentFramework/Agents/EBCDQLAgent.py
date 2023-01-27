from ExperimentFramework.Agent import AgentFactory, Agent
from Common.Types import State, Action, ActionSet
from typing import List, Tuple
import random
from collections import defaultdict
from copy import copy
import numpy as np


class EBCDQLAgentFactory(AgentFactory):

    def makeAgent(self, environmentInfo):
        return EBCDQLAgent(self.agentParameters, self.centralLearner, environmentInfo)

class EBCDQLAgent(Agent):

    def __init__(self, parameters, centralLearner, environmentInfo) -> None:
        self.epsilon = parameters["epsilon"]
        self.policyEpsilon = parameters["policyEpsilon"]
        self.beta = parameters["beta"]
        self.gamma = parameters["gamma"]
        self.rho = parameters["rho"]
        self.possibleActions = environmentInfo["actionSpace"]
        self.currentState = tuple(environmentInfo["observation"]) if environmentInfo["observation"] is not None else None
        self.obervationSpace = environmentInfo["observationSpace"]
        self.q = defaultdict(self.zerosReturn)
        self.L = 0
        super().__init__(parameters, centralLearner, environmentInfo)

    def zerosReturn(self):
        return np.zeros(self.possibleActions.n)

    def step(self, observation: State, reward: float) -> Action:
        self.lastState = copy(self.currentState)
        self.currentState = tuple(observation)
        self.lastReward = reward
        self.generateNextAction()
        tdError = reward+self.gamma*np.max(self.q[self.currentState]) - self.q[self.lastState][self.lastAction]
        self.L = (1-self.beta)*self.L + self.beta*np.abs(tdError)
        if np.abs(tdError) >= max(self.rho*self.L, self.epsilon):
            self.sendMessage((self.lastState, self.lastAction, reward, self.currentState, self.currentAction))

    def sendMessage(self, message):
        self.centralLearner.recieveMessage(self.getId(), message)
        self.lastMessage = message

    def recieveMessage(self, message):
        self.q = copy(message)
    
    def getAction(self) -> Action:
        return self.currentAction

    def nextEpisode(self, state) -> None:
        return super().nextEpisode(tuple(state))

    def generateNextAction(self) -> None:
        # with probability epsilon return a random action to explore the environment
        self.lastAction = self.currentAction
        if np.random.random() < self.epsilon:
            self.currentAction =  self.possibleActions.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            self.currentAction = int(np.argmax(self.q[self.currentState]))

        # actionValues = {action:self.q[(state, action)] for state, action in self.q.keys() if state == self.currentState}
        # bestAction = max(actionValues, key= lambda key: actionValues[key])
        # m = len(self.possibleActions)
        # weights = [self.epsilon/m if i != bestAction else self.epsilon/m + 1 - self.epsilon for i in range(m)]
        # self.lastAction = self.currentAction
        # self.currentAction = random.choices(self.possibleActions, weights=weights)[0]
        # print(f"Action values: {actionValues}, Best action: {bestAction}, Actual action: {self.currentAction}")

    def logStep(self):
        data = {
            "lastState": copy(self.lastState),
            "lastAction": copy(self.lastAction),
            "reward": copy(float(self.lastReward)),
            "currentState": copy(self.currentState),
            "currentState": copy(self.currentAction),
            "?message": copy(self.lastMessage)
        }
        self.lastMessage = None
        return data

