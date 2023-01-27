from ExperimentFramework.Algorithm import Algorithm
from ExperimentFramework.Agents.DAVIAAgent import DAVIAAgent
from ExperimentFramework.CentralLearners.DAVIALearner import DAVIALearnerFactory

class DAVIA(Algorithm):
    name = "DAVIA"
    
    def __init__(self, parameters):
        self.agentTypes = [DAVIAAgent]
        self.parameters = parameters
        self.parameters["centralLearnerParameters"]["epsilon"] = self.parameters["epsilon"]
        self.parameters["agentParameters"][0]["epsilon"] = self.parameters["epsilon"]
        self.centralLearnerFactory = DAVIALearnerFactory(self.parameters["centralLearnerParameters"])
