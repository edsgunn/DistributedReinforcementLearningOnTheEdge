from ExperimentFramework.Environments.GymEnvWrapper import GymEnv
import gymnasium as gym

class BlackJack(GymEnv):
    name = "Blackjack-v1"

    def __init__(self, parameters, contingentFactory):
        if "render_mode" in parameters.keys():
            self.environment = gym.make(BlackJack.name, render_mode=parameters["render_mode"])
        else:  
            self.environment = gym.make(BlackJack.name)
        super().__init__(parameters, contingentFactory)