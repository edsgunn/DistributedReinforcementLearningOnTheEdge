from SingleAgentTests.Agents.TD0 import TD0
from SingleAgentTests.Environments.SimpleGrid import SimpleGrid
from SingleAgentTests.Universe import Universe

environment = SimpleGrid(5,5,(4,4))
initailState = environment.getObservableState()
possibleAction = environment.getPossibleActions()
allStateActions = environment.getAllPossibleStateActions()
agent = TD0(0.9, 0.3, 0.9, initailState, possibleAction, allStateActions)
universe = Universe(environment, agent)
universe.trainMany(10, SimpleGrid, 5,5,(4,4))
