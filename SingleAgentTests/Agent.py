from argparse import Action
from SingleAgentTests.Types import ActionSet, State
from abc import ABC

class Agent(ABC):

    def __init__(self) -> None:
        pass

    def step(self, observableState: State, possibleActions: ActionSet, reward: float) -> None:
        pass

    def getAction(self) -> Action:
        pass