from opponents.random_opponent import RandomOpponent
from ppo import PPO
from ourhexgame.ourhexenv import OurHexGame
from network import PolicyValueNetwork
import os


def train_random_sparse(hyperparameters):
    """
    Train both models for a completely random opponent.
    """
    env = OurHexGame(board_size=11, render_mode="human", sparse_flag=True)

    hyperparameters = {
        **hyperparameters,
        "break_after_x_continuous_win_percent": 90,  # We want to train until 95% win rate against a random opponent
        "opponent": RandomOpponent(),
    }

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

    current_directory = os.path.dirname(os.path.abspath(__file__))

    print("Training against completely random opponent")
    player_1_model.learn(total_timesteps=-1)
    player_1_model.save_model(path=f"{current_directory}/checkpoints/")

    player_2_model.learn(total_timesteps=-1)
    player_2_model.save_model(path=f"{current_directory}/checkpoints/")
