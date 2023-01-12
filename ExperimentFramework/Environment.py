from typing import List, Tuple
from Common.Types import State, Action, ActionSet
from ExperimentFramework.Agent import Agent
from ExperimentFramework.CentralLearner import CentralLearner

class EnvironmentFactory:
    def __init__(self, environment, environmentParameters, contingentFactory):
        self.environmentParameters = environmentParameters
        self.typeOfEnvironment = environment
        self.contingentFactory = contingentFactory

    def makeEnvironments(self, numAgents):
        return [self.typeOfEnvironment(self.environmentParameters, self.contingentFactory) for _ in range(numAgents)]


class Environment:
    name = None

    def __init__(self, parameters, contingentFactory):
        self.parameters = parameters
        self.contingentFactory = contingentFactory
        self.agents = contingentFactory.makeContingent(self.getEnvironmentInfo())

    def getEnvironmentInfo(self):
        pass

    def getObservableState(self) -> State:
        pass

    def getPossibleActions(self) -> ActionSet:
        pass

    def getAllPossibleStateActions(self) -> List[Tuple[State, Action]]:
        pass 

    def step(self) -> bool:
        pass

    def getReward(self) -> float:
        pass

    def nextEpisode(self) -> None:
        pass

    @classmethod
    def getName(cls):
        return cls.name
