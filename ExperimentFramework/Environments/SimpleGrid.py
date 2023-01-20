from typing import List
from ExperimentFramework.Environment import Environment
from Common.Types import Action, ActionSet, State
from ExperimentFramework.Agent import Agent
from ExperimentFramework.CentralLearner import CentralLearner
from gymnasium.spaces import Tuple, Discrete

class SimpleGrid(Environment):
    name = "SimpleGrid"
    
    def __init__(self, parameters, contingentFactory):
        self.height = parameters["height"]
        self.width = parameters["width"]
        self.terminal = parameters["terminal"]
        self.observationSpace = Tuple((Discrete(self.width), Discrete(self.height)))
        self.agentPosition = (0,0)
        self.possibleActions = Discrete(5) #[L,R,U,D,S]
        super().__init__(parameters, contingentFactory)

    def getEnvironmentInfo(self):
        return {"actionSpace": self.possibleActions, "observation":self.getObservableState(), "observationSpace": self.observationSpace}

    def getObservableState(self) -> State:
        return self.agentPosition

    def getPossibleActions(self) -> ActionSet:
        return self.possibleActions

    def getAllPossibleStateActions(self):
        return [((x, y), action) for action in self.possibleActions for x in range(self.width) for y in range(self.height)] 

    def step(self) -> bool:
        if self.agentPosition == self.terminal:
            return False

        action = self.agents[0].getAction()
        if action == 0:
            self.agentPosition = (max(0, self.agentPosition[0]-1), self.agentPosition[1])
        elif action == 1:
            self.agentPosition = (min(self.width-1, self.agentPosition[0]+1), self.agentPosition[1])
        elif action == 2:
            self.agentPosition = (self.agentPosition[0], min(self.height-1, self.agentPosition[1]+1))
        elif action == 3:
            self.agentPosition = (self.agentPosition[0], max(0, self.agentPosition[1]-1))
        elif action == 4:
            pass
        
        self.agents[0].step(self.getObservableState(), self.getReward())
        if self.agentPosition == self.terminal:
            self.running = False
            return False
        else:
            return True

    def nextEpisode(self) -> None:
        self.running = True
        self.agentPosition = (0,0)
        self.agents[0].nextEpisode(self.getObservableState())

    def getReward(self) -> float:
        if self.agentPosition == self.terminal:
            return 0
        else:
            return -1
