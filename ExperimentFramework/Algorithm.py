
class ContingentFactory:

    def __init__(self, typesOfAgent, agentParameters, centralLearner, logger):
        self.typesOfAgent = typesOfAgent
        self.agentParameters = agentParameters
        self.centralLearner = centralLearner
        self.logger = logger

    def makeContingent(self, environmentInfo):
        contingent = []
        for typeOfAgent, agentParameters in zip(self.typesOfAgent, self.agentParameters):
            contingent.append(typeOfAgent(agentParameters, self.centralLearner, environmentInfo))
            self.logger.addAgent(contingent[-1])
        return contingent

class Algorithm:
    name = None

    def __init__(self, parameters):
        self.agentTypes = None
        self.parameters = parameters
        self.centralLearnerFactory = None
        self.agentFactoryType = None

    def makeContingentFactory(self, centralLearner, logger):
        return ContingentFactory(self.agentTypes, self.parameters["agentParameters"], centralLearner, logger)

    def makeCentralLearner(self):
        return self.centralLearnerFactory.makeCentralLearner()

    @classmethod
    def getName(cls):
        return cls.name