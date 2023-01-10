from typing import List, Tuple
from Common.Types import State, Action, ActionSet
from DistributedTests.Agent import Agent
from DistributedTests.CentralLearner import CentralLearner
from abc import ABC

class EnvironmentFactory:
    
    def makeEnvironments():
        pass

class Environment(ABC):


    def getObservableState(self) -> State:
        pass

    def getPossibleActions(self) -> ActionSet:
        pass

    def getAllPossibleStateActions(self) -> List[Tuple[State, Action]]:
        pass 

    def step(self, action: Action) -> bool:
        pass

    def getReward(self) -> float:
        pass

    def nextEpisode(self) -> None:
        pass

class SingleAgentEnvironment(Environment):
    
    def __init__(self, centralLearner: CentralLearner, agent: Agent, *agentArgs) -> None:
        self.agent = agent(*agentArgs)
        self.agent.addCentralLearner(centralLearner)

class MultiAgentEnvironement(Environment):
    pass
