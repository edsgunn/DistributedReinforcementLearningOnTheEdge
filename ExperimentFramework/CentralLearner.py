
class CentralLearnerFactory:

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

    def getV2D(self):
        v = []
        for stateAction in self.q.keys():
            x = stateAction[0][0]
            y = stateAction[0][1]
            while len(v) - 1 < x:
                v.append([])
            while len(v[x]) - 1 < y:
                v[x].append(None)
            if v[x][y] is not None:
                v[x][y] = max(v[x][y], self.q[stateAction])  
            else:
                v[x][y] = self.q[stateAction]   
        return v

    def printV2D(self):
        text = "V:\n"
        v = self.getV2D()
        for row in v:
            for entry in row:
                text += "{:.2f} ".format(entry)
            text += "\n"
        print(text)