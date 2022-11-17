from typing import cast
from SingleAgentTests.Agent import Agent
from Common.Types import State, Action, ActionSet


class Human(Agent):

    def step(self, observableState: State, possibleActions: ActionSet, reward: float) -> None:
        print(f"State: {observableState}")
        print(f"Possible actions: {possibleActions}")
        print(f"Reward: {reward}")

    def getAction(self) -> Action:
        action = input()
        return cast(Action, int(action))