from ExperimentFramework.Algorithm import Algorithm
from ExperimentFramework.Agents.EEBCDQLAgent import EEBCDQLAgent
from ExperimentFramework.CentralLearners.EEBCDQLLearner import EEBCDQLLearnerFactory

class EEBCDQL(Algorithm):
    name = "EEBCDQL"
    
    def __init__(self, parameters):
        self.agentTypes = [EEBCDQLAgent]
        self.parameters = parameters
        self.centralLearnerFactory = EEBCDQLLearnerFactory(self.parameters["centralLearnerParameters"])
