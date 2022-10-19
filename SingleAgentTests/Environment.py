from Types import State

class Environment:

    def __init__(self) -> None:
        self.currentState = self.getInitialState()

    def getInitialState() -> State:
        return 1