from ExperimentFramework.Algorithm import Algorithm
from ExperimentFramework.Agents.ESAgent import ESAgent
from ExperimentFramework.CentralLearners.ESLearner import EvolutionSamplerFactory

class EvolutionStrategies(Algorithm):
    name = "EvolutionSampling"
    
    def __init__(self, parameters):
        self.agentTypes = [ESAgent]
        self.parameters = parameters
        self.centralLearnerFactory = EvolutionSamplerFactory(self.parameters["centralLearnerParameters"])
