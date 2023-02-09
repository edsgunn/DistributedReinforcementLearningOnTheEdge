from ExperimentFramework.Environments.GymEnvWrapper import GymEnv
import gymnasium as gym
from ExperimentFramework.Environment import Feature
import numpy as np

class CliffWalkingFeature(Feature):
    def __init__(self):
        self.nrows = 3
        self.featureLength = 4*12
        # state = current_row * nrows + current_col

    def __call__(self, state, action):
        vec = np.zeros([self.featureLength ,1])
        j = state % self.nrows
        i = (state - j) // self.nrows 
        #0,1,2,3 -> U,R,D,L
        if action == 0:
            i = min(self.nrows-1, i+1)
        elif action == 1:
            j = min(self.nrows-1, j+1)
            
        elif action == 2:
            i = max(0, i-1)
        elif action == 3:
            j = max(0, j-1)

        vec[self.nrows*i+j,0] +=1
        return vec

class CliffWalking(GymEnv):
    name = "CliffWalking-v0"

    def __init__(self, parameters, contingentFactory):
        if "render_mode" in parameters.keys():
            self.environment = gym.make(CliffWalking.name, render_mode=parameters["render_mode"])
        else:  
            self.environment = gym.make(CliffWalking.name)
        self.feature = CliffWalkingFeature()
        super().__init__(parameters, contingentFactory)

    def nextEpisode(self) -> None:
        if "visualisationThreshold" in self.parameters.keys() and self.episode >= self.parameters["visualisationThreshold"]:
            self.environment = gym.make(CliffWalking.name, render_mode="human")
        super().nextEpisode()