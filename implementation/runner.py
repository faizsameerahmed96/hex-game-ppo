from time import sleep
from ourhexenv import OurHexGame
from agent_group8.g08agent import G08Agent
import random
from opponents.smart_random import SmartRandomOpponent

env = OurHexGame(board_size=11)
env.reset()

# player 1
g08agent_p1 = G08Agent(env, player="player_1", model_path="./agent_group8/model/sparse")
g08agent_p2 = G08Agent(env, player="player_2", model_path="./agent_group8/model/dense_general")

smart_agent_player_id = random.choice(env.agents)

done = False
while True:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            env.reset()
            continue

        if agent == "player_1":
            action = g08agent_p1.select_action(
                observation, reward, termination, truncation, info
            )
        else:
            action = g08agent_p2.select_action(
                observation, reward, termination, truncation, info
            )

        env.step(action)
        env.render()
