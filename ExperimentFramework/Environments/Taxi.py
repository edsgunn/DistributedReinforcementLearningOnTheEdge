from ExperimentFramework.Environments.GymEnvWrapper import GymEnv
import gymnasium as gym

class Taxi(GymEnv):
    name = "Taxi-v3"

    def __init__(self, parameters, contingentFactory):
        if "render_mode" in parameters.keys():
            self.environment = gym.make(Taxi.name, render_mode=parameters["render_mode"])
        else:  
            self.environment = gym.make(Taxi.name)
        super().__init__(parameters, contingentFactory)