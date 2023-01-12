from ExperimentFramework.Algorithm import Algorithm
from ExperimentFramework.Agents.DumbAgent import DumbAgent
from ExperimentFramework.CentralLearners.DQLearner import DQLearnerFactory

class DQL(Algorithm):
    name = "DQL"
    
    def __init__(self, parameters):
        self.agentTypes = [DumbAgent]
        self.parameters = parameters
        self.centralLearnerFactory = DQLearnerFactory(self.parameters["centralLearnerParameters"])
