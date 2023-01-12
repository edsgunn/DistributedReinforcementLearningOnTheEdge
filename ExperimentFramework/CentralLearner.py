
class CentralLearnerFactory:

    def __init__(self, parameters):
        self.parameters = parameters

    def makeCentralLearner():
        pass

class CentralLearner:

    def __init__(self) -> None:
        self.agents = []

    def step(self):
        pass

    def addAgent(self, agent):
        self.agents.append(agent)

    def recieveMessage(self, message):
        pass

    def sendMessage(self, message):
        for agent in self.agents:
            agent.recieveMessage(message)
