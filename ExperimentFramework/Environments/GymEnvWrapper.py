from ExperimentFramework.Environment import Environment
from Common.Types import State, Action, ActionSet
from typing import List, Tuple
import gymnasium as gym

class GymEnv(Environment):
    name = "gymEnv"

    def __init__(self, parameters, agentFactory):
        if "render_mode" in parameters.keys():
            self.environment = gym.make(GymEnv.name, render_mode=parameters["render_mode"])
        else:  
            self.environment = gym.make(GymEnv.name)
        self.currentObservation = None
        self.currentReward = None
        self.parameters = parameters
        self.agentFactory = agentFactory
        self.agents = agentFactory.makeAgents()
        self.running = False

    def getObservableState(self) -> State:
        return self.currentObservation

    def getPossibleActions(self) -> ActionSet:
        return self.environment.action_space

    def getAllPossibleStateActions(self) -> List[Tuple[State, Action]]:
        pass 

    def step(self, action: Action) -> bool:
        if self.running:
            self.currentObservation, self.currentReward, terminated, truncated, _ = self.environment.step(action)
            self.running = self.running | terminated or truncated

    def getReward(self) -> float:
        return self.currentReward

    def nextEpisode(self) -> None:
        self.running = True
        self.currentObservation, _ = self.environment.reset()