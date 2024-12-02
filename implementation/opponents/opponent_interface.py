class OpponentInterface:
    """
    We will use this class as a base class for all the opponents we will implement.
    """

    def __init__(self, name):
        self.name = name

    def get_action(self, observation):
        raise NotImplementedError("This method should be implemented by the subclass")
    
    def reset(self):
        raise NotImplementedError("This method should be implemented by the subclass")
