from Common.Types import ActionSet, State
from DistributedTests.CentralLearner import CentralLearner
from Common.Types import Action
from uuid import uuid4

class AgentFactory:

    def __init__(self, agentParameters, centralLearner, environmentInfo):
        self.agentParameters = agentParameters
        self.centralLearner = centralLearner
        
    def makeAgent():
        pass


class Agent:

    def __init__(self, parameters, centralLearner, environmentInfo):
        self.id = uuid4()
        self.currentAction = None
        self.lastAction = None
        self.lastMessage = None
        self.lastReward = None
        self.parameters = parameters
        self.centralLearner = centralLearner
        self.centralLearner.addAgent(self)

    def getId(self):
        return self.id

    def step(self, observation: State, reward: float) -> Action:
        pass
    
    def getAction(self) -> Action:
        return self.currentAction

    def generateNextAction(self) -> None:
        pass

    def sendMessage(self, message):
        self.lastMessage = message
        self.centralLearner.recieveMessage(self.getId(), message)

    def recieveMessage(self, message):
        pass

    def nextEpisode(self, state) -> None:
        self.lastState = None
        self.lastAction = None
        self.currentState = state
        self.generateNextAction()
