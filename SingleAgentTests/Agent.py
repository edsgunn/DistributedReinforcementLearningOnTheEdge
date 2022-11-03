from argparse import Action
from SingleAgentTests.Types import ActionSet, State
from abc import ABC

class Agent(ABC):

    def __init__(self) -> None:
        pass

    def step(self, observableState: State, possibleActions: ActionSet, reward: float) -> None:
        pass

    def nextEpisode(self) -> None:
        pass
    
    def getAction(self) -> Action:
        pass

    def generateNextAction(self) -> None:
        pass


class OnlineAgent(Agent):

    def __init__(self):
        self.currentAction = None
        self.lastAction = None
        self.generateNextAction()

    def step(self, observableState: State, possibleActions: ActionSet, reward: float):
        print(f"State: {observableState}")
        print(f"Possible actions: {possibleActions}")
        print(f"Reward: {reward}")
        self.lastState = self.currentState
        self.currentState = observableState
        self.generateNextAction()

    def getAction(self) -> Action:
        return self.currentAction

    def nextEpisode(self, state) -> None:
        self.lastState = None
        self.lastAction = None
        self.currentState = state
        self.generateNextAction()

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

class OfflineAgent(Agent):

    def __init__(self):
        pass