import random

class MyDumbAgent:
    def __init__(self):
        pass

    def select_action(self, observation):
        # Selecting a random action from available cells
        available_actions = [i for i in range(observation.size) if observation.flatten()[i] == 0]
        action = random.choice(available_actions)
        return action

class MyABitSmarterAgent:
    def __init__(self):
        pass

    def select_action(self, observation):
        #Simply using a heuristic to prioritize center cells and connecting cells
        available_actions = [i for i in range(observation.size) if observation.flatten()[i] == 0]
        
        #Demonstrating that this agent will try to play in the center of the board or nearby
        center = (len(observation) * len(observation[0])) // 2
        if center in available_actions:
            return center
        return random.choice(available_actions)
