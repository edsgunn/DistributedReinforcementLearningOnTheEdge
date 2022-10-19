from SingleAgentTests.Types import Action, ActionSet
from Types import State
from abc import ABC

class Environment(ABC):

    def __init__(self) -> None:
        pass

    def getObservableState(self) -> State:
        pass

    def getPossibleActions(self) -> ActionSet:
        pass

    def step(self, action: Action) -> bool:
        pass

    def getReward(self) -> float:
        pass
