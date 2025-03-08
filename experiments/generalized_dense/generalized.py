from opponents.duel_opponent import DuelOpponent
from opponents.smart_random import SmartRandomOpponent
from ppo import PPO
from ourhexgame.ourhexenv import OurHexGame
from network import PolicyValueNetwork
import os



CENTER_WEIGHT = 1 # Somewhat in the middle but also not too much


def train_duel_dense(hyperparameters):
    """
    Train both models against each other alternatively
    """
    env = OurHexGame(board_size=11, render_mode="human", sparse_flag=False)

    hyperparameters = {
        **hyperparameters,
        "break_after_x_continuous_win_percent": 75,
        "render_every_x_iterations": 20,
        "gamma": 0.6,
        "n_updates_per_iteration": 25,
        "episodes_per_batch": 50,
        "lr": 3e-5,
        "clip": 0.2,
        "step_reward_multiplier": 0.9,
    }

    current_directory = os.path.dirname(os.path.abspath(__file__))


    while True:
        player_1_model = PPO(
            policy_class=PolicyValueNetwork,
            env=env,
            **{**hyperparameters, "current_agent_player": "player_1",}
        )
        player_1_model.load_model_for("player_1", get_latest_model_path_for_player("player_1"))
        player_2_model = PPO(
            policy_class=PolicyValueNetwork,
            env=env,
            **{**hyperparameters, "current_agent_player": "player_2",}
        )
        player_2_model.load_model_for("player_2", get_latest_model_path_for_player("player_2"))

        player_1_model.opponent = DuelOpponent(player="player_2", env=env, actor=player_2_model.actor)
        player_2_model.opponent = DuelOpponent(player="player_1", env=env, actor=player_1_model.actor)


        player_1_model.learn(total_timesteps=-1)
        player_1_model.save_model(path=f"{current_directory}/checkpoints/")

        player_2_model.learn(total_timesteps=-1)
        player_2_model.save_model(path=f"{current_directory}/checkpoints/")


def get_latest_model_path_for_player(player):
    """
    Get the latest model for the player
    """
    path = f"/Users/faizahmed/Documents/SJSU/Reinforcement Learning/pa5/experiments/generalized_dense/checkpoints/{player}/"
    dirs = os.listdir(path)
    dirs.sort()
    return f"{path}/{dirs[-1]}"