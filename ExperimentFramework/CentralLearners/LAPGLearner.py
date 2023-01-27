from ExperimentFramework.CentralLearner import CentralLearnerFactory, CentralLearner
import numpy as np
from copy import copy
import gymnasium.spaces.utils as ut

class LAPGLearnerFactory(CentralLearnerFactory):

    def __init__(self, parameters):
        self.parameters = parameters

    def makeCentralLearner(self):
        return LAPGLearner(self.parameters)

class LAPGLearner(CentralLearner):

    def __init__(self, parameters) -> None:
        self.parameters = parameters
        self.inputSize = None
        self.hiddenSize = parameters["hiddenSize"]
        self.alpha = parameters["alpha"]
        self.outputSize = None
        self.numParams = None
        self.weights = None
        self.lastGrad = None
        self.updates = []
        super().__init__()

    def step(self):
        pass

    def nextEpisode(self, environmentInfo):
        if self.inputSize is None or self.outputSize is None or self.numParams is None or self.weights is None:
            self.inputSize = len(ut.flatten(environmentInfo["observationSpace"], environmentInfo["observationSpace"].sample()))
            self.outputSize = len(ut.flatten(environmentInfo["actionSpace"], environmentInfo["actionSpace"].sample()))
            self.numParams = self.inputSize*self.hiddenSize + self.inputSize + self.hiddenSize*self.hiddenSize + self.hiddenSize + self.hiddenSize*self.hiddenSize + self.hiddenSize + self.hiddenSize*self.outputSize + self.outputSize
            self.weights = np.zeros([self.numParams,1])
        print(self.lastGrad)
        print(np.sum(self.updates))
        print(self.weights)
        print(self.alpha*(self.lastGrad + np.sum(self.updates) if self.lastGrad is not None else np.sum(self.updates)))
        self.weights += self.alpha*(self.lastGrad + np.sum(self.updates) if self.lastGrad is not None else np.sum(self.updates))
        self.broadcastMessage(self.weights)
        self.updates = []

    def recieveMessage(self, agentId, message):
        self.updates.append(message)
    
    def logStep(self):
        data = {"?message": copy(self.lastMessage), "?weights": copy(self.weights)}
        self.lastMessage = None
        return data