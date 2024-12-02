from ourhexenv import OurHexGame
from g08agent import G08Agent
import random
from opponents.smart_random import SmartRandomOpponent

env = OurHexGame(board_size=11)
env.reset()

# player 1
g08agent = G08Agent(env, player="player_1")
smart_rand_opp = SmartRandomOpponent(player="player_2", center_weight=5)

smart_agent_player_id = random.choice(env.agents)

done = False
while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            break

        
        if agent == 'player_1':
            action = g08agent.select_action(observation, reward, termination, truncation, info)
        else:
            action = smart_rand_opp.get_action(observation)

        env.step(action)
        env.render()