from argparse import Action
from SingleAgentTests.Types import ActionSet, State
from abc import ABC

class Agent(ABC):

    def step(self, observableState: State, possibleActions: ActionSet, reward: float) -> None:
        pass

    def nextEpisode(self) -> None:
        pass
    
    def getAction(self) -> Action:
        pass