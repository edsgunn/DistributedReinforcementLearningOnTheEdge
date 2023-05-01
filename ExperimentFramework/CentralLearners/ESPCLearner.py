from ExperimentFramework.CentralLearner import CentralLearnerFactory, CentralLearner
import numpy as np
from copy import copy
import gymnasium.spaces.utils as ut

class ESPCLearnerFactory(CentralLearnerFactory):

    def __init__(self, parameters):
        self.parameters = parameters

    def makeCentralLearner(self):
        return ESPCLearner(self.parameters)

class ESPCLearner(CentralLearner):

    def __init__(self, parameters) -> None:
        self.rewards = {}
        self.parameters = parameters
        self.rng = np.random.default_rng(self.parameters["seed"])
        self.inputSize = None
        self.hiddenSize = parameters["hiddenSize"]
        self.outputSize = None
        self.numParams = None
        self.agentGenerators = {}
        self.weights = None
        self.episode = 0
        self.defualtRewards = None
        self.numMessages = 0
        super().__init__()

    def step(self):
        pass

    def nextEpisode(self, environmentInfo):
        if self.inputSize is None or self.outputSize is None or self.numParams is None:
            self.inputSize = len(ut.flatten(environmentInfo["observationSpace"], environmentInfo["observationSpace"].sample()))
            self.outputSize = len(ut.flatten(environmentInfo["actionSpace"], environmentInfo["actionSpace"].sample()))
            self.numParams = self.inputSize*self.hiddenSize + self.inputSize + self.hiddenSize*self.hiddenSize + self.hiddenSize + self.hiddenSize*self.hiddenSize + self.hiddenSize + self.hiddenSize*self.outputSize + self.outputSize
            self.defualtRewards = {agent.getId(): 0 for agent in self.agents}
            self.rewards = copy(self.defualtRewards)
        if self.weights is None:
            self.weights = self.rng.multivariate_normal(np.zeros(self.numParams), np.eye(self.numParams))
            self.velocity = np.zeros(self.weights.shape)
        else:
            vec_sum = sum([reward*self.agentGenerators[agent].multivariate_normal(np.zeros(self.numParams), np.eye(self.numParams)) for agent, reward in self.rewards.items()])
            if self.numMessages > 0:
                grad = self.parameters["vstep"]*(1/(self.parameters["sigma"]*self.numMessages))*vec_sum - self.parameters["l2"] * self.weights
                self.velocity = self.parameters["gamma"]*self.velocity + grad
                self.weights += self.velocity
        self.rewards = copy(self.defualtRewards)
        self.broadcastMessage(self.weights)
        self.numMessages = 0
        self.episode += 1

    def addAgent(self, agent):
        super().addAgent(agent)
        seed = self.rng.integers(0,self.parameters["maxSeedInt"])
        self.agentGenerators[agent.getId()] = np.random.default_rng(seed)

    def recieveMessage(self, agentId, message):
        self.rewards[agentId] = message
        self.numMessages +=1
    
    def logStep(self):
        data = {"?message": copy(self.lastMessage), "?weights": copy(self.weights)}
        self.lastMessage = None
        return data