from typing import Any, List, Optional, Tuple, Type
from SingleAgentTests.Environment import Environment
from SingleAgentTests.Agent import Agent

class Universe:

    def __init__(self, environments: List[Environment], centralLearner: CentralLearner) -> None:
        self.environment = environment
        self.agent = agent
        self.centralLearner = centralLearner
        self.history = []

    def step(self) -> None:
        # print(f"Step: {self.time}\n---------------------------------")
        action = self.agent.getAction()
        self.running = self.environment.step(action)
        observableState = self.environment.getObservableState()
        possibleActions = self.environment.getPossibleActions()
        reward = self.environment.getReward()
        self.agent.step(observableState, possibleActions, reward)
        
        self.history.append((self.history[-1][2], action, observableState, reward, None if self.running else self.time))
        self.time += 1

    def start(self):
        self.running = True
        self.time: int = 0
        observableState = self.environment.getObservableState()
        possibleActions = self.environment.getPossibleActions()
        self.history.append((None, None, observableState, None, None))
        # print(f"Observable state: {observableState}")
        # print(f"Possible actions: {possibleActions}")
        while self.running:
            self.step()
            if self.time > 1000000:
                break

    def trainMany(self, iterations: int, environment: Type[Environment], *args: Any):
        for _ in range(iterations):
            self.agent.printV2D()
            self.environment = environment(*args)
            state = self.environment.getObservableState()
            self.agent.nextEpisode(state)
            self.start()

    def getHistory(self) -> List[Tuple[Optional[int],...]]:
        return self.history
        

