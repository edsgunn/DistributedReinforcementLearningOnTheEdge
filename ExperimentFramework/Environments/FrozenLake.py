from ExperimentFramework.Environments.GymEnvWrapper import GymEnv
import gymnasium as gym
from ExperimentFramework.Environment import Feature
import numpy as np

class FrozenLakeFeature(Feature):
    def __init__(self, isSlippery):
        self.nrows = 4
        self.isSlippery = isSlippery
        self.featureLength = self.nrows**2

    def __call__(self, state, action):
        vec = np.zeros([self.nrows**2,1])
        j = state % self.nrows
        i = (state - j) // self.nrows 
        #0,1,2,3 -> L,D,R,U
        if action == 0:
            j = max(0, j-1)
        elif action == 1:
            i = max(0, i-1)
        elif action == 2:
            i = min(self.nrows-1, i+1)
        elif action == 3:
            j = min(self.nrows-1, j+1)
            

        if self.isSlippery:
            vec[self.nrows*i+j,0] +=1/3
            i1 = i
            i2 = i
            j1 = j
            j2 = j
            if action == 0:
                i1 = max(0, i-1)
                i2 = min(self.nrows-1,i+1)
            elif action == 1:
                j1 = max(0, j-1)
                j2 = min(self.nrows-1,j+1)
            elif action == 2:
                j1 = max(0, j-1)
                j2 = min(self.nrows-1,j+1)
            elif action == 3:
                i1 = max(0, i-1)
                i2 = min(self.nrows-1,i+1)
                
            elif action == 4:
                pass

            vec[self.nrows*i1+j1,0] +=1/3
            vec[self.nrows*i2+j2,0] +=1/3


        else:
            vec[self.nrows*i+j,0] +=1
        # vec[-1,0] = 1
        return vec


class FrozenLake(GymEnv):
    name = "FrozenLake-v1"

    def __init__(self, parameters, contingentFactory):
        if "isSlippery" not in parameters.keys():
            self.isSlippery = False
        else:
            self.isSlippery = parameters["isSlippery"]
        if "render_mode" in parameters.keys():
            self.environment = gym.make(FrozenLake.name, map_name="4x4", render_mode=parameters["render_mode"], is_slippery=self.isSlippery)
        else:  
            self.environment = gym.make(FrozenLake.name, map_name="4x4", is_slippery=self.isSlippery)
        self.feature = FrozenLakeFeature(self.isSlippery)
        super().__init__(parameters, contingentFactory)