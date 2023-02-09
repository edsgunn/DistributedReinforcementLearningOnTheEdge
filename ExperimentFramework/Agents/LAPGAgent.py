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
from torch.nn.utils import parameters_to_vector


class LAPGAgentFactory(AgentFactory):

    def makeAgent(self, environmentInfo):
        return LAPGAgent(self.agentParameters, self.centralLearner, environmentInfo)

class LAPGAgent(Agent):

    def __init__(self, parameters, centralLearner, environmentInfo) -> None:
        self.gamma = parameters["gamma"]
        self.N = parameters["N"]
        self.D = parameters["D"]
        self.M = parameters["M"]
        self.eta = parameters["eta"]
        self.alpha = parameters["alpha"]
        self.possibleActions = environmentInfo["actionSpace"]
        self.observationSpace = environmentInfo["observationSpace"]
        self.currentState = environmentInfo["observation"]
        self.lastWeights = None
        self.weightsUpdates = []
        self.inputSize = len(ut.flatten(self.observationSpace, self.observationSpace.sample()))
        self.hiddenSize = parameters["hiddenSize"]
        self.outputSize = len(ut.flatten(self.possibleActions, self.possibleActions.sample()))
        self.trajectoryHistory = []
        self.currentTrajectory = []
        self.numParams = self.inputSize*self.hiddenSize + self.inputSize + self.hiddenSize*self.hiddenSize + self.hiddenSize + self.hiddenSize*self.hiddenSize + self.hiddenSize + self.hiddenSize*self.outputSize + self.outputSize
        self.model = nn.Sequential(
                        nn.Linear(self.inputSize, self.hiddenSize),
                        nn.Tanh(),
                        nn.Linear(self.hiddenSize, self.hiddenSize),
                        nn.Tanh(),
                        nn.Linear(self.hiddenSize, self.hiddenSize),
                        nn.Tanh(),
                        nn.Linear(self.hiddenSize, self.outputSize),
                        nn.Softmax(dim=0))
        super().__init__(parameters, centralLearner, environmentInfo)

    def calculateGrad(self):
        grad = None
        currentWeights = self.model.state_dict()
        for episode in self.trajectoryHistory:
            for t, (_, reward, _) in enumerate(episode):
                stepGrad = None
                for (state, action), _, weights in episode[:t+1]:
                    if weights is not None:
                        self.model.load_state_dict(weights)
                    flatState = ut.flatten(self.observationSpace, state).astype(np.float32)
                    flatState = torch.from_numpy(flatState)
                    probabilities = self.model(flatState)
                    indicator = torch.zeros([1,self.outputSize])
                    indicator[0,action] = 1
                    loss = torch.log(torch.matmul(indicator,probabilities.view([self.outputSize,1])))
                    if stepGrad is None:
                        stepGrad = torch.autograd.grad(loss, self.model.parameters())
                    else:
                        for p,q in zip(stepGrad,torch.autograd.grad(loss, self.model.parameters())):
                            p.data += q.data
                if grad is None:
                    grad = stepGrad
                    for p in grad:
                        p *= (self.gamma**t)*reward
                else:
                    for p,q in zip(grad, stepGrad):
                        p.data += q.data*(self.gamma**t)*reward
        self.model.load_state_dict(currentWeights)
        return grad

    def step(self, observation: State, reward: float) -> Action:
        self.lastState = copy(self.currentState)
        self.lastAction = copy(self.currentAction)
        self.currentState = observation
        self.lastReward = reward
        self.currentTrajectory.append(((self.lastState,self.lastAction),reward, self.lastWeights))
        self.lastWeights = None
        self.generateNextAction()

    def nextEpisode(self, state) -> None:
        if len(self.trajectoryHistory) >= self.N:
            self.trajectoryHistory.pop(0)
        if self.currentTrajectory:
            self.trajectoryHistory.append(self.currentTrajectory)
        self.currentTrajectory = []
        grad = self.calculateGrad()
        if self.trajectoryHistory:
            # print(grad)
            # print(torch.norm(parameters_to_vector(grad)), ((self.eta/(self.alpha**2 * self.M**2))*sum(self.weightsUpdates) if self.weightsUpdates else 0))
            if torch.norm(parameters_to_vector(grad)) >= ((self.eta/(self.alpha**2 * self.M**2))*sum(self.weightsUpdates) if self.weightsUpdates else 0):
                self.sendMessage(grad)
        super().nextEpisode(state)

    def recieveMessage(self, message):
        if len(self.weightsUpdates) >= self.D:
            self.weightsUpdates.pop(0)
        self.weightsUpdates.append(torch.norm(parameters_to_vector(message) - parameters_to_vector(self.model.parameters())))
        self.lastWeights = self.model.state_dict()
        for p,q in zip(message,self.model.parameters()):
            q.data = p


    def generateNextAction(self) -> None:
        flatState = ut.flatten(self.observationSpace, self.currentState).astype(np.float32)
        flatState = torch.from_numpy(flatState)
        probabilities = self.model(flatState)
        self.currentAction = int(torch.multinomial(probabilities, 1)[0])

    def logStep(self):
        data = {
            "lastState": copy(self.lastState),
            "lastAction": copy(self.lastAction),
            "reward": copy(float(self.lastReward)),
            "currentState": copy(self.currentState),
            "currentState": copy(self.currentAction),
            "?message": copy(self.lastMessage)
        }
        self.lastReward = 0
        self.lastMessage = None
        return data

