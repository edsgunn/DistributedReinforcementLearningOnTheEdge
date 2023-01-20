from ExperimentFramework.Environments.GymEnvWrapper import GymEnv
import gymnasium as gym

class LunarLander(GymEnv):
    name = "LunarLander-v2"

    def __init__(self, parameters, contingentFactory):
        if "render_mode" in parameters.keys():
            self.environment = gym.make(LunarLander.name, render_mode=parameters["render_mode"])
        else:  
            self.environment = gym.make(LunarLander.name)
        super().__init__(parameters, contingentFactory)