from ExperimentFramework.Algorithm import Algorithm
from ExperimentFramework.Agents.EBCDQLAgent import EBCDQLAgent
from ExperimentFramework.CentralLearners.EBCDQLLearner import EBCDQLLearnerFactory

class EBCDQL(Algorithm):
    name = "EBCDQL"
    
    def __init__(self, parameters):
        self.agentTypes = [EBCDQLAgent]
        self.parameters = parameters
        self.centralLearnerFactory = EBCDQLLearnerFactory(self.parameters["centralLearnerParameters"])
