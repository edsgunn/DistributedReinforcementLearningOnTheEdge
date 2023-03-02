from ExperimentFramework.Algorithm import Algorithm, ContingentFactory
from ExperimentFramework.Agents.ESPCAgent import ESPCAgent
from ExperimentFramework.CentralLearners.ESPCLearner import ESPCLearnerFactory
import numpy as np
from copy import deepcopy

class ESPCContingentFactory(ContingentFactory):

    def __init__(self, typesOfAgent, agentParameters, centralLearner, logger):
        self.typesOfAgent = typesOfAgent
        self.agentParameters = agentParameters
        self.centralLearner = centralLearner
        self.logger = logger
        self.rng = np.random.default_rng(agentParameters[0]["seed"])

    def makeContingent(self, environmentInfo):
        self.agentParameters[0]["seed"] = self.rng.integers(0,self.agentParameters[0]["maxSeedInt"])
        contingent = []
        for typeOfAgent, agentParameters in zip(self.typesOfAgent, self.agentParameters):
            contingent.append(typeOfAgent(agentParameters, self.centralLearner, environmentInfo))
            self.logger.addAgent(contingent[-1])
        return contingent

class ESPC(Algorithm):
    name = "ESPC"
    
    def __init__(self, parameters):
        self.agentTypes = [ESAgent]
        self.parameters = parameters
        self.parameters["centralLearnerParameters"]["seed"] = self.parameters["seed"]
        self.parameters["centralLearnerParameters"]["sigma"] = self.parameters["sigma"]
        self.parameters["centralLearnerParameters"]["hiddenSize"] = self.parameters["hiddenSize"]
        self.parameters["centralLearnerParameters"]["maxSeedInt"] = self.parameters["maxSeedInt"]
        self.parameters["centralLearnerParameters"]["optimalReward"] = self.parameters["optimalReward"]
        self.parameters["centralLearnerParameters"]["alpha"] = self.parameters["alpha"]
        self.parameters["centralLearnerParameters"]["beta"] = self.parameters["beta"]
        self.parameters["agentParameters"][0]["seed"] = self.parameters["seed"]
        self.parameters["agentParameters"][0]["sigma"] = self.parameters["sigma"]
        self.parameters["agentParameters"][0]["hiddenSize"] = self.parameters["hiddenSize"]
        self.parameters["agentParameters"][0]["alpha"] = self.parameters["alpha"]
        self.parameters["agentParameters"][0]["beta"] = self.parameters["beta"]
        self.parameters["agentParameters"][0]["maxSeedInt"] = self.parameters["maxSeedInt"]
        self.parameters["agentParameters"][0]["optimalReward"] = self.parameters["optimalReward"]

        self.centralLearnerFactory = ESLearnerFactory(self.parameters["centralLearnerParameters"])

    def makeContingentFactory(self, centralLearner, logger):
        return ESContingentFactory(self.agentTypes, deepcopy(self.parameters["agentParameters"]), centralLearner, logger)
