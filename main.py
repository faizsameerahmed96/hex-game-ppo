import sys

import gymnasium as gym
import torch

from arguments import get_args
from eval_policy import eval_policy
from network import PolicyValueNetwork
from ourhexgame.ourhexenv import OurHexGame
from ppo import PPO
import os


def get_latest_model_for_player(player):
    path = f"./checkpoints/{player}/actor"
    files = os.listdir(path)
    files.sort()
    return path + "/" + files[-1]

def main():
    hyperparameters = {
        "episodes_per_batch": 20,
        "timesteps_per_batch": 2048,
        "max_timesteps_per_episode": 300,
        "gamma": 0.6,
        "n_updates_per_iteration": 20,
        "lr": 3e-4,
        "clip": 0.2,
        "save_freq": 200,
        "render": True,
        "render_every_i": 10,
        "break_after_x_win_percent": 80,
        "train_against_opponent": False,
        "render_every_x_iterations": 5,
    }

    # Delete old models

    env = OurHexGame(board_size=11, render_mode="human", sparse_flag=False)

    player_1_model = PPO(
        policy_class=PolicyValueNetwork,
        env=env,
        **{**hyperparameters, "current_agent_player": "player_1"}
    )
    player_2_model = PPO(
        policy_class=PolicyValueNetwork,
        env=env,
        **{**hyperparameters, "current_agent_player": "player_2"}
    )

    # First generalize both models on their own before starting the duel training loop
    # print("Training for general information")
    # player_1_model.learn(total_timesteps=-1)
    # player_2_model.learn(total_timesteps=-1)

    
    print("Dueling training start")
    # Reusming training
    player_1_model.load_model_for("player_1", get_latest_model_for_player("player_1"))
    player_2_model.load_model_for("player_2", get_latest_model_for_player("player_2"))

    # Duel training loop
    player_1_model.train_against_opponent = True
    player_2_model.train_against_opponent = True
    player_1_model.load_model_for("player_2", get_latest_model_for_player("player_2"))
    player_2_model.load_model_for("player_1", get_latest_model_for_player("player_1"))
    player_1_model.break_after_x_win_percent = 80
    player_2_model.break_after_x_win_percent = 80

    while True:
        player_2_model.learn(total_timesteps=-1)
        player_1_model.load_model_for("player_2", get_latest_model_for_player("player_2"))

        player_1_model.learn(total_timesteps=-1)
        player_2_model.load_model_for("player_1", get_latest_model_for_player("player_1"))

        delete_old_models("./checkpoints/player_1/actor")
        delete_old_models("./checkpoints/player_1/critic")
        delete_old_models("./checkpoints/player_2/actor")
        delete_old_models("./checkpoints/player_2/critic")


def delete_old_models(path):
    files = os.listdir(path)
    files.sort()
    if len(files) > 10:
        for i in range(len(files) - 2):
            os.remove(path + "/" + files[i])


if __name__ == "__main__":
    main()
