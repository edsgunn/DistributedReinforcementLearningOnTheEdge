from ExperimentFramework.Agent import AgentFactory, Agent
from Common.Types import State, Action, ActionSet
from typing import List, Tuple
import random
from collections import defaultdict
from copy import copy
import numpy as np


class DumbAgentFactory(AgentFactory):

    def makeAgent(self, environmentInfo):
        return DumbAgent(self.agentParameters, self.centralLearner, environmentInfo)

class DumbAgent(Agent):

    def __init__(self, parameters, centralLearner, environmentInfo) -> None:
        self.epsilon = parameters["epsilon"]
        self.possibleActions = environmentInfo["actionSpace"]
        self.currentState = environmentInfo["observation"]
        self.q = defaultdict(lambda: np.zeros(self.possibleActions.n))
        super().__init__(parameters, centralLearner)

    def step(self, observation: State, reward: float) -> Action:
        self.lastState = copy(self.currentState)
        self.currentState = observation
        self.lastReward = reward
        self.generateNextAction()
        self.sendMessage((self.lastState, self.lastAction, reward, self.currentState, self.currentAction))

    def sendMessage(self, message):
        self.centralLearner.recieveMessage(message)

    def recieveMessage(self, message):
        self.q = copy(message)
    
    def getAction(self) -> Action:
        return self.currentAction

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
            "lastState": str(self.lastState),
            "lastAction": str(self.lastAction),
            "reward": str(self.lastReward),
            "currentState": str(self.currentState),
            "currentState": str(self.currentAction),
            "?message": str(self.lastMessage)
        }
        self.lastMessage = None
        return data

