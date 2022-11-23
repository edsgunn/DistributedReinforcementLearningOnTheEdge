from typing import Any, List, Optional, Tuple, Type
from DistributedTests.Environment import Environment
from DistributedTests.CentralLearner import CentralLearner

class Universe:

    def __init__(self, environments: List[Environment], centralLearner: CentralLearner) -> None:
        self.environments = environments
        self.centralLearner = centralLearner
        self.time = None
        self.running = None

    def step(self) -> None:
        for environment in self.environments:
            environment.step()
        self.centralLearner.step()
        
        self.time += 1

    def start(self):
        self.running = True
        self.time: int = 0
        while self.running:
            self.step()
            if self.time > 1000000:
                break

    def trainMany(self, iterations: int, environment: Type[Environment], *args: Any):
        for _ in range(iterations):
            for enviroment in self.environments:
                environment.nextEpisode()
            self.start()

        

