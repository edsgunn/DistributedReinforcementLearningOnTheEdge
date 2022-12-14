from typing import List, Tuple
from Common.Types import State, Action, ActionSet
from abc import ABC

class Environment(ABC):

    def __init__(self) -> None:
        pass

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
