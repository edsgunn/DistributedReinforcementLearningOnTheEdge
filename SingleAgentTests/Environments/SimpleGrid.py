from typing import List, Tuple
from SingleAgentTests.Environment import Environment
from SingleAgentTests.Types import Action, ActionSet, State
import numpy as np

class SimpleGrid(Environment):

    def __init__(self, width: int, height: int, terminal: Tuple[int, int]) -> None:
        self.grid = [[0 for _ in range(height)] for _ in range(width)]
        self.terminal = terminal
        self.agentPosition = (0,0)
        self.possibleActions = [0,1,2,3,4] #[L,R,U,D,S]

    def getObservableState(self) -> State:
        return self.agentPosition

    def getPossibleActions(self) -> ActionSet:
        return self.possibleActions

    def getAllPossibleStateActions(self) -> List[Tuple[State, Action]]:
        return [((x, y), action) for action in self.possibleActions for x in range(len(self.grid)) for y in range(len(self.grid[0]))] 

    def step(self, action: Action) -> bool:
        if action == 0:
            self.agentPosition = (max(0, self.agentPosition[0]-1), self.agentPosition[1])
        elif action == 1:
            self.agentPosition = (min(len(self.grid)-1, self.agentPosition[0]+1), self.agentPosition[1])
        elif action == 2:
            self.agentPosition = (self.agentPosition[0], min(len(self.grid[0])-1, self.agentPosition[1]+1))
        elif action == 3:
            self.agentPosition = (self.agentPosition[0], max(0, self.agentPosition[1]-1))
        elif action == 4:
            pass

        if self.agentPosition == self.terminal:
            return False
        else:
            return True


    def getReward(self) -> float:
        if self.agentPosition == self.terminal:
            return 0
        else:
            return -1
