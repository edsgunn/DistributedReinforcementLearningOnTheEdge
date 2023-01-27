from ExperimentFramework.Algorithm import Algorithm
from ExperimentFramework.Agents.LAPGAgent import LAPGAgent
from ExperimentFramework.CentralLearners.LAPGLearner import LAPGLearnerFactory

class LAPG(Algorithm):
    name = "LAPG"
    
    def __init__(self, parameters):
        self.agentTypes = [LAPGAgent]
        self.parameters = parameters
        self.parameters["centralLearnerParameters"]["alpha"] = self.parameters["alpha"]
        self.parameters["centralLearnerParameters"]["hiddenSize"] = self.parameters["hiddenSize"]
        self.parameters["agentParameters"][0]["alpha"] = self.parameters["alpha"]
        self.parameters["agentParameters"][0]["hiddenSize"] = self.parameters["hiddenSize"]
        self.centralLearnerFactory = LAPGLearnerFactory(self.parameters["centralLearnerParameters"])
