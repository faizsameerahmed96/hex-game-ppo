import os
from experiments.smart_random_1.smart_random import train_smart_random
from experiments.smart_random_1_5.smart_random import train_smart_random_1_5
from network import PolicyValueNetwork
from opponents.random_opponent import RandomOpponent
from ourhexgame.ourhexenv import OurHexGame
from ppo import PPO
from utils import get_latest_model_for_player

from experiments.full_random.full_random import train_random


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
    # train_random(hyperparameters)

    # Train both models for a smart random opponent
    # train_smart_random(hyperparameters)

    # Train a more focused smart random opponent
    train_smart_random_1_5(hyperparameters)

    return

    env = OurHexGame(board_size=11, render_mode="human", sparse_flag=False)

    player_1_model = PPO(
        policy_class=PolicyValueNetwork,
        env=env,
        **{**hyperparameters, "current_agent_player": "player_1"},
    )
    player_2_model = PPO(
        policy_class=PolicyValueNetwork,
        env=env,
        **{**hyperparameters, "current_agent_player": "player_2"},
    )

    # First generalize both models on their own before starting the duel training loop
    # print("Training for general information")
    # player_1_model.load_model_for("player_1", get_latest_model_for_player("player_1"))
    player_1_model.learn(total_timesteps=-1)
    # player_2_model.learn(total_timesteps=-1)

    return
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
        player_1_model.load_model_for(
            "player_2", get_latest_model_for_player("player_2")
        )

        player_1_model.learn(total_timesteps=-1)
        player_2_model.load_model_for(
            "player_1", get_latest_model_for_player("player_1")
        )

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
