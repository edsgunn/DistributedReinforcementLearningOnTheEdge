from ExperimentFramework.Environment import Environment
from Common.Types import State, Action, ActionSet
from typing import List, Tuple
import gymnasium as gym

class GymEnv(Environment):
    name = "gymEnv"

    def __init__(self, parameters, contingentFactory):
        self.currentObservation = None
        self.currentReward = None
        self.running = False
        self.episode = 0
        super().__init__(parameters, contingentFactory)
    
    def getEnvironmentInfo(self):
        return {"actionSpace": self.environment.action_space, "observation":self.currentObservation, "observationSpace": self.environment.observation_space, "feature": self.feature}

    def getObservableState(self) -> State:
        return self.currentObservation

    def getPossibleActions(self) -> ActionSet:
        return self.environment.action_space

    def step(self) -> bool:
        if self.running:
            for agent in self.agents:
                self.currentObservation, self.currentReward, terminated, truncated, _ = self.environment.step(agent.getAction())
                agent.step(self.currentObservation, self.currentReward)
            self.running = not (terminated or truncated)
        return self.running

    def getReward(self) -> float:
        return self.currentReward

    def nextEpisode(self) -> None:
        self.running = True
        self.currentObservation, _ = self.environment.reset()
        for agent in self.agents:
            agent.nextEpisode(self.currentObservation)
        self.episode += 1