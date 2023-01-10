class Logger:

    def __init__(self, centralLearner, parameters):
        self.data = {"trialId": 1, "parameters": parameters, "algorithms": {}}
        self.algorithm = None
        self.numAgents = None
        self.environment = None
        self.episodeNum = None

        self.centralLearner = centralLearner
        self.agentLoggers = []

    def setAlgorithm(self, algorithm: str, parameters):
        self.algorithm = algorithm
        self.data["algorithms"][self.algorithm]["parameters"] = parameters

    def setNumberOfAgents(self, numAgents: str):
        self.numAgents = numAgents

    def setEnvironment(self, environment: str, parameters):
        self.environment = environment
        self.data["algorithms"][self.algorithm][self.numAgents][self.environment]["parameters"] = parameters

    def setEpisode(self, episodeNum: int):
        self.episodeNum = episodeNum

    def logStep(self, stepNumber, step):
        step = {"centralLearner": self.centralLearner.logStep()}
        for agentLogger in self.agentLoggers:
            agentId, agentData = agentLogger.logStep()
            step[agentId] = agentData
        self.data["algorithms"][self.algorithm][self.numAgents][self.environment][stepNumber] = step

class AgentLogger:

    def _init__(self, agent):
        self.agent = agent

    def logStep(self):
        return self.agent.getId(), self.agent.logStep()
