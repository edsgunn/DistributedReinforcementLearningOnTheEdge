class Logger:

    def __init__(self, parameters):
        self.data = {"trialId": 1, "parameters": parameters, "algorithms": {}}
        self.algorithm = None
        self.numAgents = None
        self.environment = None
        self.episodeNum = None

        self.centralLearner = None
        self.agentLoggers = None

    def addCentralLearner(self, centralLearner):
        self.centralLearner = centralLearner

    def addAgent(self, agent):
        self.agentLoggers.append(AgentLogger(agent))

    def setAlgorithm(self, algorithm: str, parameters):
        self.algorithm = algorithm
        self.data["algorithms"][self.algorithm] = {}
        self.data["algorithms"][self.algorithm]["parameters"] = parameters

    def setNumberOfAgents(self, numAgents: str):
        self.numAgents = numAgents
        self.data["algorithms"][self.algorithm][self.numAgents] = {}

    def setEnvironment(self, environment: str, parameters):
        self.environment = environment
        self.data["algorithms"][self.algorithm][self.numAgents][self.environment] = {}
        self.data["algorithms"][self.algorithm][self.numAgents][self.environment]["parameters"] = parameters
        self.centralLearner = None
        self.agentLoggers = []

    def setEpisode(self, episodeNum: int):
        self.episodeNum = episodeNum
        self.data["algorithms"][self.algorithm][self.numAgents][self.environment][self.episodeNum] = {}

    def logStep(self, stepNumber):
        step = {"centralLearner": self.centralLearner.logStep()}
        for agentLogger in self.agentLoggers:
            agentId, agentData = agentLogger.logStep()
            step[agentId] = agentData
        self.data["algorithms"][self.algorithm][self.numAgents][self.environment][self.episodeNum][stepNumber] = step

class AgentLogger:

    def __init__(self, agent):
        self.agent = agent

    def logStep(self):
        return self.agent.getId(), self.agent.logStep()
