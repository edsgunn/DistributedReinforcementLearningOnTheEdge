from SingleAgentTests.Environment import Environment
from SingleAgentTests.Agent import Agent
from abc import ABC

class Universe(ABC):

    def __init__(self, environment: Environment, agent: Agent) -> None:
        self.time: int = 0
        self.environment = environment
        self.agent = agent
        self.running = True

    def step(self) -> None:
        print(f"Step: {self.time}\n---------------------------------")
        action = self.agent.getAction()
        self.running = self.environment.step(action)
        observableState = self.environment.getObservableState()
        possibleActions = self.environment.getPossibleActions()
        reward = self.environment.getReward()
        self.agent.step(observableState, possibleActions, reward)
        
        self.time += 1

    def start(self):
        observableState = self.environment.getObservableState()
        possibleActions = self.environment.getPossibleActions()
        print(f"Observable state: {observableState}")
        print(f"Possible actions: {possibleActions}")
        while self.running:
            self.step()
