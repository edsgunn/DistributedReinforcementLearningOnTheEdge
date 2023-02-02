from ExperimentFramework.CentralLearner import CentralLearnerFactory, CentralLearner
import numpy as np
from copy import copy
import gymnasium.spaces.utils as ut
import torch


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
        self.lastMessage = None
        super().__init__()

    def step(self):
        pass

    def nextEpisode(self, environmentInfo):
        if self.inputSize is None or self.outputSize is None or self.numParams is None:
            self.inputSize = len(ut.flatten(environmentInfo["observationSpace"], environmentInfo["observationSpace"].sample()))
            self.outputSize = len(ut.flatten(environmentInfo["actionSpace"], environmentInfo["actionSpace"].sample()))
            self.numParams = self.inputSize*self.hiddenSize + self.inputSize + self.hiddenSize*self.hiddenSize + self.hiddenSize + self.hiddenSize*self.hiddenSize + self.hiddenSize + self.hiddenSize*self.outputSize + self.outputSize
        if self.updates:
            if self.weights is None:
                self.weights = copy(self.updates[0])
                if self.lastGrad is None:
                    updates = [self.weights]
                    i = 1
                else:
                    updates = [self.weights, self.lastGrad]
                    i = 2
                updates.extend([update for update in self.updates])
                for vars in zip(*updates):
                    for x in vars:
                        cumGrads = sum([p.data  for p in x[i:]])
                        if i==1:
                            x[0].data = self.alpha*(cumGrads)
                        else:
                            x[0].data = self.alpha*(x[1]+cumGrads)
            else:
                if self.lastGrad is None:
                    updates = [self.weights]
                    i = 1
                else:
                    updates = [self.weights, self.lastGrad]
                    i = 2
                updates.extend([update for update in self.updates])
                for vars in zip(*updates):
                    for x in vars:
                        cumGrads = sum([p.data  for p in x[i:]])
                        if i==1:
                            x[0].data += self.alpha*(cumGrads)
                        else:
                            x[0].data += self.alpha*(x[1]+cumGrads)

            self.broadcastMessage(self.weights)
            self.updates = []

    def recieveMessage(self, agentId, message):
        self.updates.append(message)
    
    def logStep(self):
        data = {"?message": copy(self.lastMessage), "?weights": copy(self.weights)}
        self.lastMessage = None
        return data