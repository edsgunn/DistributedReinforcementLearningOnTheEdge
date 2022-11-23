from typing import List
from DistributedTests.Agent import Agent

class CentralLearner:

    def __init__(self) -> None:
        self.agents: List[Agent] = []

    def step(self):
        pass

    def addAgent(self, agent: Agent):
        self.agents.append(agent)

    def recieveMessage(self, message):
        pass

    def sendMessage(self, message):
        for agent in self.agents:
            agent.recieveMessage(message)