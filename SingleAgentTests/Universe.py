from Environment import Environment
from Agent import Agent

class Universe:

    def __init__(self, environment: Environment, agent: Agent):
        self.time: int = 0
        self.enviroment = environment
