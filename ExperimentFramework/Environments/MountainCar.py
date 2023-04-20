from ExperimentFramework.Environments.GymEnvWrapper import GymEnv
import gymnasium as gym
from ExperimentFramework.Environment import Feature
import numpy as np

class MountainCar(GymEnv):
    name = "MountainCar-v0"

    def __init__(self, parameters, contingentFactory):
        if "render_mode" in parameters.keys():
            self.environment = gym.make(MountainCar.name, render_mode=parameters["render_mode"])
        else:  
            self.environment = gym.make(MountainCar.name)
        self.feature = None
        super().__init__(parameters, contingentFactory)

    def nextEpisode(self) -> None:
        if "visualisationThreshold" in self.parameters.keys() and self.episode >= self.parameters["visualisationThreshold"]:
            self.environment = gym.make(MountainCar.name, render_mode="human")
        super().nextEpisode()