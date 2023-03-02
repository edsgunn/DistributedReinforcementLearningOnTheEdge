from ExperimentFramework.Algorithm import Algorithm
from ExperimentFramework.Agents.EDumbAgent import EDumbAgent
from ExperimentFramework.CentralLearners.EDQLearner import EDQLearnerFactory

class EDQL(Algorithm):
    name = "EDQL"
    
    def __init__(self, parameters):
        self.agentTypes = [EDumbAgent]
        self.parameters = parameters
        self.centralLearnerFactory = EDQLearnerFactory(self.parameters["centralLearnerParameters"])
