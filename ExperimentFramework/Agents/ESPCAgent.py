from ExperimentFramework.Agent import AgentFactory, Agent
from Common.Types import State, Action, ActionSet
from typing import List, Tuple
import random
from collections import OrderedDict
from copy import copy
import numpy as np
import torch.nn as nn
import torch
import gymnasium.spaces.utils as ut


class ESPCAgentFactory(AgentFactory):

    def makeAgent(self, environmentInfo):
        return ESPCAgent(self.agentParameters, self.centralLearner, environmentInfo)

class ESPCAgent(Agent):

    def __init__(self, parameters, centralLearner, environmentInfo) -> None:
        self.possibleActions = environmentInfo["actionSpace"]
        self.observationSpace = environmentInfo["observationSpace"]
        self.currentState = environmentInfo["observation"]
        self.rgn = np.random.default_rng(parameters["seed"])
        self.weights = None
        self.inputSize = len(ut.flatten(self.observationSpace, self.observationSpace.sample()))
        self.hiddenSize = parameters["hiddenSize"]
        self.sigma = parameters["sigma"]
        self.alpha = parameters["alpha"]
        self.beta = parameters["beta"]
        self.outputSize = len(ut.flatten(self.possibleActions, self.possibleActions.sample()))
        self.numParams = self.inputSize*self.hiddenSize + self.inputSize + self.hiddenSize*self.hiddenSize + self.hiddenSize + self.hiddenSize*self.hiddenSize + self.hiddenSize + self.hiddenSize*self.outputSize + self.outputSize
        self.mu = np.zeros(self.numParams)
        self.cov = (parameters["sigma"]**2)*np.eye(self.numParams)
        self.model = nn.Sequential(
                        nn.Linear(self.inputSize, self.hiddenSize),
                        nn.Tanh(),
                        nn.Linear(self.hiddenSize, self.hiddenSize),
                        nn.Tanh(),
                        nn.Linear(self.hiddenSize, self.hiddenSize),
                        nn.Tanh(),
                        nn.Linear(self.hiddenSize, self.outputSize),
                        nn.Softmax(dim=0))
        self.totalReward = 0
        self.bestReward = parameters["optimalReward"]
        self.episode = 0
        super().__init__(parameters, centralLearner, environmentInfo)

    def arrangeParameters(self, vector):
        params = OrderedDict()
        currentParams = self.model.state_dict()
        elNum = 0
        for name, tensor in currentParams.items():
            num = torch.numel(tensor)
            params[name] = torch.from_numpy(vector[elNum:num]).view(tensor.size())
        return params

    def step(self, observation: State, reward: float) -> Action:
        # print(f"Reward: {reward}")
        self.lastState = copy(self.currentState)
        self.currentState = observation
        self.lastReward = reward
        self.totalReward += reward
        self.generateNextAction()

    def nextEpisode(self, state) -> None:
        super().nextEpisode(state)
        self.bestReward = max(self.bestReward, self.totalReward)
        if random.random() < self.alpha*(self.bestreward-self.totalReward)/self.bestReward*np.exp(-self.beta*self.episode):
            self.sendMessage(self.totalReward)
        self.totalReward = 0
        self.episode += 1



    def sendMessage(self, message):
        self.centralLearner.recieveMessage(self.getId(), message)
        self.lastMessage = message

    def recieveMessage(self, message):
        self.weights = copy(message) + self.sigma*self.rgn.multivariate_normal(self.mu, self.cov, method="cholesky")
        self.model.load_state_dict(self.arrangeParameters(self.weights))

    def getAction(self) -> Action:
        return self.currentAction

    def generateNextAction(self) -> None:
        flatState = ut.flatten(self.observationSpace, self.currentState).astype(np.float32)
        flatState = torch.from_numpy(flatState)
        probabilities = self.model(flatState)
        self.currentAction = int(torch.argmax(probabilities)) #int(torch.multinomial(probabilities, 1)[0])
        # print(self.currentState, probabilities)

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
        self.lastReward = 0
        return data

