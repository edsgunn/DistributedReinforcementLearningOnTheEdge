from ExperimentFramework.CentralLearner import CentralLearnerFactory, CentralLearner
from typing import List, Tuple
from Common.Types import State, Action
from collections import defaultdict
import numpy as np
import json

class DQLearnerFactory(CentralLearnerFactory):

    def __init__(self, parameters):
        self.parameters = parameters

    def makeCentralLearner(self):
        return DQLearner(self.parameters)

class DQLearner(CentralLearner):

    def __init__(self, parameters) -> None:
        self.updates = defaultdict(self.listReturn)
        self.alpha = parameters["alpha"]
        self.gamma = parameters["gamma"]
        self.actionSpace = None
        self.q = defaultdict(self.zerosReturn)
        super().__init__()

    def listReturn(self):
        return []

    def zerosReturn(self):
        return np.zeros(self.actionSpace.n)

    def step(self):
        for stateAction, update in self.updates.items():
            self.q[stateAction[0]][stateAction[1]] += self.alpha*np.mean(update)
        self.updates = defaultdict(self.listReturn)
        self.sendMessage(self.q)

    def nextEpisode(self, environmentInfo):
        self.actionSpace = environmentInfo["actionSpace"]
        self.updates = defaultdict(self.listReturn)
        self.q = defaultdict(self.zerosReturn)

    def recieveMessage(self, message):
        self.updates[(message[0],message[1])].append(message[2]+self.gamma*max(self.q[message[3]]) - self.q[message[0]][message[1]])

    def getQ(self):
        return self.q
    
    def logStep(self):
        data = {"message": self.lastMessage, "valueFunction": self.q}
        self.lastMessage = None
        return data

    