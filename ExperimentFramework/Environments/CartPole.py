from ExperimentFramework.Environments.GymEnvWrapper import GymEnv
import gymnasium as gym

class CartPole(GymEnv):
    name = "CartPole-v1"

    def __init__(self, parameters, contingentFactory):
        if "render_mode" in parameters.keys():
            self.environment = gym.make(CartPole.name, render_mode=parameters["render_mode"])
        else:  
            self.environment = gym.make(CartPole.name)
        super().__init__(parameters, contingentFactory)