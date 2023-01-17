from ExperimentFramework.Environments.GymEnvWrapper import GymEnv
import gymnasium as gym

class FrozenLake(GymEnv):
    name = "FrozenLake-v1"

    def __init__(self, parameters, contingentFactory):
        if "render_mode" in parameters.keys():
            self.environment = gym.make(FrozenLake.name, render_mode=parameters["render_mode"])
        else:  
            self.environment = gym.make(FrozenLake.name)
        super().__init__(parameters, contingentFactory)