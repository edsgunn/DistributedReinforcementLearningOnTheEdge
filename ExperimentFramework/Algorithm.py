
class ContingentFactory:

    def __init__(self, typesOfAgent, agentParameters, centralLearner):
        self.typesOfAgent = typesOfAgent
        self.agentParameters = agentParameters
        self.centralLearner = centralLearner

    def makeContingent(self, environmentInfo):
        return [typeOfAgent(agentParameters, self.centralLearner, environmentInfo) for typeOfAgent, agentParameters in zip(self.typesOfAgent, self.agentParameters)]

class Algorithm:
    name = None

    def __init__(self, parameters):
        self.agentTypes = None
        self.parameters = parameters
        self.centralLearnerFactory = None
        self.agentFactoryType = None

    def makeContingentFactory(self, centralLearner):
        return ContingentFactory(self.agentTypes, self.parameters["agentParameters"], centralLearner)

    def makeCentralLearner(self):
        return self.centralLearnerFactory.makeCentralLearner()

    @classmethod
    def getName(cls):
        return cls.name