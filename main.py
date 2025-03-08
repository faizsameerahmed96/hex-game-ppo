import os

from experiments.full_random.full_random import train_random
from experiments.full_random_sparse.full_random import train_random_sparse
from experiments.generalized_dense.generalized import train_duel_dense
from experiments.smart_random_1.smart_random import train_smart_random
from experiments.smart_random_1_5.smart_random import train_smart_random_1_5
from experiments.smart_random_1_5_sparse.smart_random import train_smart_random_1_5
from experiments.smart_random_1_sparse.smart_random import train_smart_random_sparse
from experiments.smart_random_2.smart_random import train_smart_random_2
from network import PolicyValueNetwork
from opponents.random_opponent import RandomOpponent
from ourhexgame.ourhexenv import OurHexGame
from ppo import PPO
from utils import get_latest_model_for_player


def main():
    hyperparameters = {
        "episodes_per_batch": 20,
        "max_timesteps_per_episode": 300,
        "gamma": 0.6,
        "n_updates_per_iteration": 15,
        "lr": 3e-5,
        "clip": 0.2,
        "save_freq": 200_000,
        "render_every_x_iterations": 100,
        "max_num_of_episodes_to_calculate_win_percent": 20,
        "break_after_x_continuous_win_percent": 101,
        "how_many_consecutive_wins_to_break": 5,
        "step_reward_multiplier": 1,
        "opponent": RandomOpponent(),
    }

    # Train both models for a completely random opponent
    train_random(hyperparameters)

    # Train both models for a smart random opponent
    train_smart_random(hyperparameters)

    # Train a more focused smart random opponent
    train_smart_random_1_5(hyperparameters)

    train_smart_random_2(hyperparameters)

    train_duel_dense(hyperparameters)

    hyperparameters = {
        "episodes_per_batch": 20,
        "max_timesteps_per_episode": 300,
        "gamma": 0.6,
        "n_updates_per_iteration": 15,
        "lr": 3e-5,
        "clip": 0.2,
        "save_freq": 200_000,
        "render_every_x_iterations": 100,
        "max_num_of_episodes_to_calculate_win_percent": 20,
        "break_after_x_continuous_win_percent": 101,
        "how_many_consecutive_wins_to_break": 5,
        "step_reward_multiplier": 1,
        "opponent": RandomOpponent(),
    }

    # SPARSE TRAINING
    train_random_sparse(hyperparameters)

    train_smart_random_sparse(hyperparameters)

    train_smart_random_1_5(hyperparameters)


def delete_old_models(path):
    files = os.listdir(path)
    files.sort()
    if len(files) > 10:
        for i in range(len(files) - 2):
            os.remove(path + "/" + files[i])


if __name__ == "__main__":
    main()
