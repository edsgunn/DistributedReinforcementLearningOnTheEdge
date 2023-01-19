from ExperimentFramework.CentralLearner import CentralLearnerFactory, CentralLearner
import numpy as np
from copy import copy
import gymnasium.spaces.utils as ut

class ESLearnerFactory(CentralLearnerFactory):

    def __init__(self, parameters):
        self.parameters = parameters

    def makeCentralLearner(self):
        return ESLearner(self.parameters)

class ESLearner(CentralLearner):

    def __init__(self, parameters) -> None:
        self.rewards = {}
        self.parameters = parameters
        self.rng = np.random.default_rng(self.parameters["seed"])
        self.inputSize = None
        self.hiddenSize = parameters["hiddenSize"]
        self.outputSize = None
        self.numParams = None

        self.weights = None
        super().__init__()

    def step(self):
        pass

    def nextEpisode(self, environmentInfo):
        if self.inputSize is None or self.outputSize is None or self.numParams is None:
            self.inputSize = len(ut.flatten(environmentInfo["observationSpace"], environmentInfo["observationSpace"].sample()))
            self.outputSize = len(ut.flatten(environmentInfo["actionSpace"], environmentInfo["actionSpace"].sample()))
            self.numParams = self.inputSize*self.hiddenSize + self.inputSize + self.hiddenSize*self.hiddenSize + self.hiddenSize + self.hiddenSize*self.hiddenSize + self.hiddenSize + self.hiddenSize*self.outputSize + self.outputSize
        if self.rewards and self.weights is not None:
            self.weights += self.parameters["alpha"]*(1/(self.parameters["sigma"]*len(self.agents)))*np.sum([reward*self.agentGenerators[agent].multivariate_normal(np.zeros(self.numParams), np.eye(self.numParams)) for agent, reward in self.rewards.items()])
            self.rewards = {}
        else:
            self.weights = self.rng.multivariate_normal(np.zeros(self.numParams), np.eye(self.numParams))
        self.broadcastMessage(self.weights)

    def addAgent(self, agent):
        super().addAgent(agent)
        self.agentGenerators[agent.getId()] = self.rng.randint(0,1000000)

    def recieveMessage(self, agentId, message):
        self.rewards[agentId] = message

    def getQ(self):
        return self.q
    
    def logStep(self):
        data = {"?message": copy(self.lastMessage), "?valueFunction": copy(self.q)}
        self.lastMessage = None
        return data