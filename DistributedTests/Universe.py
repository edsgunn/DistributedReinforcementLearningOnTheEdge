from typing import Any, List, Optional, Tuple, Type
from DistributedTests.Environment import Environment
from DistributedTests.CentralLearner import CentralLearner

class Universe:

    def __init__(self, environments: List[Environment], centralLearner: CentralLearner) -> None:
        self.environments = environments
        self.centralLearner = centralLearner
        self.time = None
        self.running = None
        self.episode = 0

    def step(self) -> None:
        self.running = False
        for environment in self.environments:
            self.running |= environment.step()
        self.centralLearner.step()
        
        self.time += 1

    def start(self):
        self.running = True
        self.time: int = 0
        # print(f"Episode: {self.episode}")
        while self.running:
            self.step()
            if self.time > 1000000:
                break
        self.episode += 1

    def trainMany(self, iterations: int):
        for _ in range(iterations):
            for enviroment in self.environments:
                enviroment.nextEpisode()
            self.start()

        

