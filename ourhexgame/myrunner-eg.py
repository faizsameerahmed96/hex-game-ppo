from ourhexenv import OurHexGame
# from gXXagent import GXXAgent
# from gYYagent import GYYAgent
import random

env = OurHexGame(board_size=11)
env.reset()

class MyDumbAgent:
    def __init__(self, game: OurHexGame):
        self.env = game

    def select_action(self, observation, reward, termination, truncation, info):
        # get indexes of observation that are 0

        obs = observation['observation']
        zero_indexes = [
            (i, j)
            for i in range(len(obs))
            for j in range(len(obs[i]))
            if obs[i][j] == 0
        ]
        row, col = random.choice(zero_indexes)
        return row * self.env.board_size + col

# player 1
gXXagent = MyDumbAgent(env)
# player 2
gYYagent = MyDumbAgent(env)

smart_agent_player_id = random.choice(env.agents)

done = False
while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            break

        
        if agent == 'player_1':
            action = gXXagent.select_action(observation, reward, termination, truncation, info)
        else:
            action = gYYagent.select_action(observation, reward, termination, truncation, info)

        env.step(action)
        env.render()
