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

class Feature:

    def __init__(self, action, numStates, numActions):
        self.action = action
        self.numStates = numStates
        self.numActions = numActions

    def __call__(self, state):
        vec = np.zeros([self.numStates+1, 1])
        for i in range(self.numActions):
            vec[self.numActions*self.action+i,1] = state[i]
            vec[-1,1] = 1
        return vec 

class DAVIAAgent(Agent):

    def __init__(self, parameters, centralLearner, environmentInfo) -> None:
        self.epsilon = parameters["epsilon"]
        self.lambd = parameters["lambda"]
        self.rho = parameters["rho"]
        self.observationSpace = environmentInfo["observationSpace"]
        self.actionSpace = environmentInfo["actionSpace"]
        numObservations = ut.flatten(self.observationSpace, self.observationSpace.sample())
        numActions = ut.flatten(self.actionSpace, self.actionSpace.sample())
        self.weights = None #Row vector for ease of use
        self.features = [Feature(action, numObservations, numActions) for action in range(numActions)]
        super().__init__(parameters, centralLearner)


    def step(self, observation: State, reward: float) -> Action:
        self.lastState = copy(self.currentState)
        self.currentState = observation
        self.lastReward = reward
        self.generateNextAction()
        if self.thresholdReached:
            self.sendMessage(self.grad)

    def sendMessage(self, message):
        self.centralLearner.recieveMessage(self.getId(), message)
        self.lastMessage = message

    def recieveMessage(self, message):
        self.weights = copy(message)
    
    def getAction(self) -> Action:
        return self.currentAction

    def generateNextAction(self) -> None:
        self.lastAction = self.currentAction
        if np.random.random() < self.epsilon:
            self.currentAction =  self.possibleActions.sample()
        else:
            self.currentAction = int(np.argmax([self.q(self.currentState, action) for action in range(self.obervationSpace.nvec)]))

    def q(self, state, action):
        return np.matmul(self.weights, self.features[action](state))

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
        return data

