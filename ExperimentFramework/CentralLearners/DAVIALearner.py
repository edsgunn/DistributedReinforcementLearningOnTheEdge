from ExperimentFramework.CentralLearner import CentralLearnerFactory, CentralLearner
from typing import List, Tuple
from Common.Types import State, Action
from collections import defaultdict
import numpy as np
from copy import copy
import gymnasium.spaces.utils as ut

class DAVIALearnerFactory(CentralLearnerFactory):

    def __init__(self, parameters):
        self.parameters = parameters

    def makeCentralLearner(self):
        return DAVIALearner(self.parameters)

class DAVIALearner(CentralLearner):

    def __init__(self, parameters) -> None:
        self.weights = None
        self.epsilon = parameters["epsilon"]
        self.actionSpace = None
        self.observationSpace = None
        self.updates = []
        super().__init__()

    def step(self):
        if self.updates:
            self.weights += self.epsilon*np.mean(self.updates)
            self.broadcastMessage(self.weights)
            self.updates = []

    def nextEpisode(self, environmentInfo):
        if any(x is None for x in [self.actionSpace, self.observationSpace, self.weights]):
            self.actionSpace = environmentInfo["actionSpace"]
            self.observationSpace = environmentInfo["observationSpace"]
            numObservations = ut.flatten(self.observationSpace, self.observationSpace.sample())
            self.weights = np.zeros([1,numObservations+1]) #Row vector for ease of use

    def recieveMessage(self, agentId, message):
        self.updates.append(message)
    
    def logStep(self):
        data = {"?message": copy(self.lastMessage), "?weights": copy(self.weights)}
        self.lastMessage = None
        return data

    