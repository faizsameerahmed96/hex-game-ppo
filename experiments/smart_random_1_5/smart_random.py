from opponents.smart_random import SmartRandomOpponent
from ppo import PPO
from ourhexgame.ourhexenv import OurHexGame
from network import PolicyValueNetwork
import os



CENTER_WEIGHT = 1.5 # more focused


def train_smart_random_1_5(hyperparameters):
    """
    Train both models for a smart random opponent.
    """
    env = OurHexGame(board_size=11, render_mode="human", sparse_flag=False)

    hyperparameters = {
        **hyperparameters,
        "break_after_x_continuous_win_percent": 90,
        "render_every_x_iterations": 20,
        "episodes_per_batch": 20,
        "clip": 0.2,
        "lr": 3e-5,
        "n_updates_per_iteration": 20,
        "gamma": 0.6,
    }

    player_1_model = PPO(
        policy_class=PolicyValueNetwork,
        env=env,
        **{**hyperparameters, "current_agent_player": "player_1", "opponent": SmartRandomOpponent(player="player_2", center_weight=CENTER_WEIGHT)}
    )
    player_1_model.load_model_for("player_1", '/Users/faizahmed/Documents/SJSU/Reinforcement Learning/pa5/experiments/smart_random_1/checkpoints/player_1/1733133180.83876')
    player_2_model = PPO(
        policy_class=PolicyValueNetwork,
        env=env,
        **{**hyperparameters, "current_agent_player": "player_2", "opponent": SmartRandomOpponent(player="player_1", center_weight=CENTER_WEIGHT)}
    )
    player_2_model.load_model_for("player_2", '/Users/faizahmed/Documents/SJSU/Reinforcement Learning/pa5/experiments/smart_random_1/checkpoints/player_2/1733133437.0088232')

    current_directory = os.path.dirname(os.path.abspath(__file__))

    print("Training against smart random opponent with center weight", CENTER_WEIGHT)

    player_1_model.learn(total_timesteps=-1)
    player_1_model.save_model(path=f"{current_directory}/checkpoints/")

    player_2_model.learn(total_timesteps=-1)
    player_2_model.save_model(path=f"{current_directory}/checkpoints/")
