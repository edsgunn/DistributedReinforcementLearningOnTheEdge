from ExperimentFramework.Agent import AgentFactory, Agent
from Common.Types import State, Action, ActionSet
from typing import List, Tuple
import random
from collections import defaultdict
from copy import copy
import numpy as np
import gymnasium.spaces.utils as ut

class DAVIAAgentFactory(AgentFactory):

    def makeAgent(self, environmentInfo):
        return DAVIAAgent(self.agentParameters, self.centralLearner, environmentInfo)


class DAVIAAgent(Agent):

    def __init__(self, parameters, centralLearner, environmentInfo) -> None:
        self.epsilon = parameters["epsilon"]
        self.lambd = parameters["lambda"]
        self.rho = parameters["rho"]
        self.gamma = parameters["gamma"]
        self.reg = parameters["reg"]
        self.N = parameters["N"]
        self.k = 0
        self.policyEpsilon = parameters["policyEpsilon"]
        self.observationSpace = environmentInfo["observationSpace"]
        self.actionSpace = environmentInfo["actionSpace"]
        self.experience = []
        self.feature = environmentInfo["feature"]
        self.weights = np.zeros([1,self.feature.len()]) #Row vector for ease of use
        self.lastWeights = copy(self.weights)
        self.lastMessage = None
        super().__init__(parameters, centralLearner, environmentInfo)


    def step(self, observation: State, reward: float) -> Action:
        self.lastState = copy(self.currentState)
        self.currentState = observation
        self.lastReward = reward
        self.generateNextAction()
        self.experience.append((self.lastState, self.lastAction, reward, self.currentState, self.currentAction))
    
    def nextEpisode(self, state) -> None:
        if self.experience:
            grad = self.calculateGrad()
            # if grad[-2] != 0:
            #     print(grad[-2], self.lastState, self.lastAction, self.lastReward, self.currentState, self.currentAction, self.experience)
            # print(self.calculateHessian().shape)
            # print(grad.shape)
            # print(np.matmul(np.matmul(np.transpose(grad), self.calculateHessian()), grad) , self.lambd/(self.rho**(max(self.N-1-self.k, 0))))
            if np.matmul(np.matmul(np.transpose(grad), self.calculateHessian()), grad) >= self.lambd/(self.rho**(max(self.N-1-self.k, 0))):
                self.sendMessage(grad)
            self.experience = []
        self.k += 1
        self.policyEpsilon = max(0.05, self.policyEpsilon - 0.01)
        super().nextEpisode(state)

    def calculateGrad(self):
        # for lastState, lastAction, reward, currentState, _ in self.experience:
            # if  (self.feature(lastState,lastAction)*(np.matmul(self.lastWeights,self.feature(lastState, lastAction)) + reward - self.gamma*max(self.q(currentState, action) for action in range(ut.flatdim(self.actionSpace)))))[-2] != 0:
                # print(lastState, lastAction, reward, currentState)
                # print((np.matmul(self.weights,self.feature(lastState, lastAction)) - reward - self.gamma*max(self.q(currentState, action) for action in range(ut.flatdim(self.actionSpace)))))

        return (1/len(self.experience))*sum(self.feature(lastState,lastAction)*(np.matmul(self.weights,self.feature(lastState, lastAction)) - reward - self.gamma*max(self.q(currentState, action) for action in range(ut.flatdim(self.actionSpace)))) for lastState, lastAction, reward, currentState, _ in self.experience) + np.transpose(self.reg*self.weights)

    def calculateHessian(self):
        return np.eye(self.feature.len()) - self.epsilon*(1/2)*(1/len(self.experience))*sum(np.matmul(self.feature(state, action), np.transpose(self.feature(state, action))) for state, action, _, _, _ in self.experience)

    def sendMessage(self, message):
        self.centralLearner.recieveMessage(self.getId(), message)
        self.lastMessage = message

    def recieveMessage(self, message):
        self.lastWeights = copy(self.weights)
        self.weights = copy(message)
    
    def getAction(self) -> Action:
        return self.currentAction

    def generateNextAction(self) -> None:
        self.lastAction = self.currentAction
        if np.random.random() < self.policyEpsilon:
            self.currentAction =  self.actionSpace.sample()
        else:
            actions = np.array([self.q(self.currentState, action) for action in range(ut.flatdim(self.actionSpace))])
            self.currentAction = int(np.random.choice(np.flatnonzero(actions == actions.max())))

    def q(self, state, action):
        return np.matmul(self.weights, self.feature(state, action))

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

