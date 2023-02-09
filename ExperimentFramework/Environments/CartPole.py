from ExperimentFramework.Environments.GymEnvWrapper import GymEnv
import gymnasium as gym
from ExperimentFramework.Environment import Feature
import math
import numpy as np

class CartPoleFeature(Feature):
    def __init__(self):
        self.featureLength = 4

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

    def __call__(self, state, action):
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        return np.array([[x], [x_dot], [theta], [theta_dot]], dtype=np.float32)

class CartPole(GymEnv):
    name = "CartPole-v1"

    def __init__(self, parameters, contingentFactory):
        if "render_mode" in parameters.keys():
            self.environment = gym.make(CartPole.name, render_mode=parameters["render_mode"])
        else:  
            self.environment = gym.make(CartPole.name)
        self.feature = CartPoleFeature()
        super().__init__(parameters, contingentFactory)

    def nextEpisode(self) -> None:
        if "visualisationThreshold" in self.parameters.keys() and self.episode >= self.parameters["visualisationThreshold"]:
            self.environment = gym.make(CartPole.name, render_mode="human")
        super().nextEpisode()