from ExperimentFramework.Environments.GymEnvWrapper import GymEnv
import gymnasium as gym

class CliffWalking(GymEnv):
    name = "CliffWalking-v0"

    def __init__(self, parameters, contingentFactory):
        if "render_mode" in parameters.keys():
            self.environment = gym.make(CliffWalking.name, render_mode=parameters["render_mode"])
        else:  
            self.environment = gym.make(CliffWalking.name)
        super().__init__(parameters, contingentFactory)