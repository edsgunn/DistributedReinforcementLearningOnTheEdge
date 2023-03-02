
class CentralLearnerFactory:

    def __init__(self, parameters):
        self.parameters = parameters

    def makeCentralLearner():
        pass

class CentralLearner:

    def __init__(self) -> None:
        self.agents = []
        self.lastMessage = None

    def step(self):
        pass

    def addAgent(self, agent):
        self.agents.append(agent)

    def recieveMessage(self, agentId, message):
        pass

    def sendMessage(self, agentId, message):
        self.lastMessage = message
        for agent in self.agents:
            if agent.getId() == agentId:
                agent.recieveMessage(message)

    def broadcastMessage(self, message):
        self.lastMessage = message
        for agent in self.agents:
            agent.recieveMessage(message)    

    def logStep():
        pass
