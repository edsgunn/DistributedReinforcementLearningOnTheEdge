from typing import List, Tuple
from SingleAgentTests.Environment import SingleAgentEnvironment
from Common.Types import Action, ActionSet, State
from DistributedTests.Agent import Agent
from DistributedTests.CentralLearner import CentralLearner

class SimpleGrid(SingleAgentEnvironment):

    def __init__(self, centralLearner: CentralLearner, width: int, height: int, terminal: Tuple[int, int], agent: Agent, *agentArgs) -> None:
        self.height = height
        self.width = width
        self.terminal = terminal
        self.agentPosition = (0,0)
        self.possibleActions = [0,1,2,3,4] #[L,R,U,D,S]
        super().__init__(centralLearner, agent, self.getObservableState(), self.getPossibleActions(), self.getAllPossibleStateActions(), *agentArgs)

    def getObservableState(self) -> State:
        return self.agentPosition

    def getPossibleActions(self) -> ActionSet:
        return self.possibleActions

    def getAllPossibleStateActions(self) -> List[Tuple[State, Action]]:
        return [((x, y), action) for action in self.possibleActions for x in range(self.width) for y in range(self.height)] 

    def step(self) -> bool:
        action = self.agent.getAction()
        if action == 0:
            self.agentPosition = (max(0, self.agentPosition[0]-1), self.agentPosition[1])
        elif action == 1:
            self.agentPosition = (min(self.width-1, self.agentPosition[0]+1), self.agentPosition[1])
        elif action == 2:
            self.agentPosition = (self.agentPosition[0], min(self.height-1, self.agentPosition[1]+1))
        elif action == 3:
            self.agentPosition = (self.agentPosition[0], max(0, self.agentPosition[1]-1))
        elif action == 4:
            pass
        
        self.agent.step(self.getObservableState(), self.getPossibleActions(), self.getReward())
        if self.agentPosition == self.terminal:
            return False
        else:
            return True

    def nextEpisode(self) -> None:
        self.agentPosition = (0,0)
        self.agent.nextEpisode()



    def getReward(self) -> float:
        if self.agentPosition == self.terminal:
            return 0
        else:
            return -1
