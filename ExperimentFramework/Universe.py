from typing import Any, Dict, List, Type
from ExperimentFramework.Environment import Environment, EnvironmentFactory
from ExperimentFramework.CentralLearner import CentralLearner, CentralLearnerFactory
from ExperimentFramework.Agent import Agent
from ExperimentFramework.Logger import Logger
class Universe:

    def __init__(self, typesOfEnvironment: List[Type[Environment]], environmentParameters, algorithms, algorithmParameters, parameters) -> None:
        self.typesOfEnvironment = typesOfEnvironment
        self.environmentParameters = environmentParameters
        self.algorithms = algorithms
        self.algorithmParameters = algorithmParameters
        self.parameters = parameters

        self.environmentFactory = None
        
        self.logger = Logger(self.parameters)
        self.algorithm = None
        self.environments = None
        self.centralLearner = None

        self.stepNumber = None
        self.running = None
        self.episode = None


    def runTrial(self):
        for algorithm, algorithmParameters in zip(self.algorithms, self.algorithmParameters):
            self.algorithm = algorithm(algorithmParameters)
            self.logger.setAlgorithm(algorithm.getName(), algorithmParameters)
            for numAgents in self.parameters["numbersOfAgents"]:
                self.logger.setNumberOfAgents(numAgents)
                for environment, environmentParameters in zip(self.typesOfEnvironment, self.environmentParameters):
                    self.logger.setEnvironment(environment.getName(), environmentParameters)
                    self.centralLearner = self.algorithm.makeCentralLearner()
                    self.logger.addCentralLearner(self.centralLearner)
                    self.environmentFactory = EnvironmentFactory(environment, environmentParameters, self.algorithm.makeContingentFactory(self.centralLearner, self.logger))
                    self.environments = self.environmentFactory.makeEnvironments(numAgents)
                    self.episode = 0
                    self.runTraining(self.parameters["numEpisodes"])

        

    def runTraining(self, iterations: int):
        for _ in range(iterations):
            for enviroment in self.environments:
                enviroment.nextEpisode()
            self.centralLearner.nextEpisode(self.environments[0].getEnvironmentInfo())
            self.start()

    def step(self) -> None:
        self.running = False
        for environment in self.environments:
            self.running |= environment.step()
        self.centralLearner.step()
        
        self.stepNumber += 1

    def start(self):
        self.running = True
        self.logger.setEpisode(self.episode)
        self.stepNumber: int = 0
        while self.running:
            self.step()
            self.logger.logStep(self.stepNumber)
            if maxSteps := self.parameters.get("maxSteps"):
                if self.stepNumber > maxSteps:
                    break
        self.episode += 1        

        

