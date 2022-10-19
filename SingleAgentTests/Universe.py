from Environment import Environment
from Agent import Agent
from abc import ABC

class Universe(ABC):

    def __init__(self, environment: Environment, agent: Agent) -> None:
        self.time: int = 0
        self.environment = environment
        self.agent = agent

    def step(self) -> None:
        action = self.agent.getAction()
        self.environment.step(action)
        observableState = self.environment.getObservableState()
        possibleActions = self.environment.getPossibleActions()
        reward = self.environment.getReward()
        self.agent.step(observableState, possibleActions, reward)

        self.time += 1
