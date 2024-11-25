"""
	This file is the executable for running PPO. It is based on this medium article: 
	https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import sys

import gymnasium as gym
import torch

from arguments import get_args
from eval_policy import eval_policy
from network import PolicyValueNetwork
from ourhexgame.ourhexenv import OurHexGame
from ppo import PPO


def main():
    hyperparameters = {
        "episodes_per_batch": 100,
        "timesteps_per_batch": 2048,
        "max_timesteps_per_episode": 300,
        "gamma": 0.6,
        "n_updates_per_iteration": 10,
        "lr": 3e-4,
        "clip": 0.2,
        "render": False,
        "render_every_i": 10,
        "break_after_x_win_percent": 80,
    }

    env = OurHexGame(board_size=11, render_mode="human", sparse_flag=False)

    player_1_model = PPO(policy_class=PolicyValueNetwork, env=env, **{**hyperparameters, 'current_agent_player': 'player_1'})
    player_2_model = PPO(policy_class=PolicyValueNetwork, env=env, **{**hyperparameters, 'current_agent_player': 'player_2'})

    while True:
        player_1_model.learn(total_timesteps=-1)
        player_2_model.load_opponent_model()
        player_2_model.learn(total_timesteps=-1)
        player_1_model.load_opponent_model()


if __name__ == "__main__":
    main()
