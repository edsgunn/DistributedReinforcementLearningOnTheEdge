from Environment import Environment
from Agent import Agent

class Universe:

    def __init__(self, environment: Environment, agent: Agent) -> None:
        self.time: int = 0
        self.environment = environment
        self.agent = agent

    def step(self) -> None:
        observableState = self.environment.getObservableState()
        possibleActions = self.environment.getPossibleActions()
        self.agent.step(observableState, possibleActions)
        action = self.agent.getAction(self.time, observableState)
        self.environment.step(action)

        self.time += 1
