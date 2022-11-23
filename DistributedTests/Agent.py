from Common.Types import ActionSet, State
from DistributedTests.CentralLearner import CentralLearner
from Common.Types import Action


class Agent:

    def __init__(self):
        self.currentAction = None
        self.lastAction = None
        self.centralLearner = None
        self.generateNextAction()

    def addCentralLearner(self, centralLearner: CentralLearner):
        self.centralLearner = centralLearner
        self.centralLearner.addAgent(self)

    def step(self, observableState: State, possibleActions: ActionSet, reward: float) -> Action:
        pass

    def nextEpisode(self) -> None:
        pass
    
    def getAction(self) -> Action:
        return self.currentAction

    def generateNextAction(self) -> None:
        pass

    def sendMessage(self, message):
        pass

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

    def nextEpisode(self, state) -> None:
        self.lastState = None
        self.lastAction = None
        self.currentState = state
        self.generateNextAction()


class OnlineAgent(Agent):

    def step(self, observableState: State, possibleActions: ActionSet, reward: float):
        print(f"State: {observableState}")
        print(f"Possible actions: {possibleActions}")
        print(f"Reward: {reward}")
        self.lastState = self.currentState
        self.currentState = observableState
        self.generateNextAction()


    


class OfflineAgent(Agent):

    def __init__(self, trainingInterval: int) -> None:
        super().__init__()
        self.trainingInterval = trainingInterval
        self.episodeNumber = 1

    def nextEpisode(self, state) -> None:
        super().nextEpisode(state)
        self.episodeNumber += 1
        print(f"Episode number: {self.episodeNumber}, Training Interval: {self.trainingInterval}")
        if self.episodeNumber % self.trainingInterval == 0:
            self.train()

    def train(self) -> None:
        pass