from opponents.smart_random import SmartRandomOpponent
from ppo import PPO
from ourhexgame.ourhexenv import OurHexGame
from network import PolicyValueNetwork
import os



CENTER_WEIGHT = 1 # Somewhat in the middle but also not too much


def train_smart_random(hyperparameters):
    """
    Train both models for a smart random opponent.
    """
    env = OurHexGame(board_size=11, render_mode="human", sparse_flag=False)

    hyperparameters = {
        **hyperparameters,
        "break_after_x_continuous_win_percent": 90,
        "render_every_x_iterations": 20,
    }

    player_1_model = PPO(
        policy_class=PolicyValueNetwork,
        env=env,
        **{**hyperparameters, "current_agent_player": "player_1", "opponent": SmartRandomOpponent(player="player_2", center_weight=CENTER_WEIGHT)}
    )
    player_1_model.load_model_for("player_1", '/Users/faizahmed/Documents/SJSU/Reinforcement Learning/pa5/experiments/full_random/checkpoints/player_1/1733112909.861575/actor.pth')
    player_2_model = PPO(
        policy_class=PolicyValueNetwork,
        env=env,
        **{**hyperparameters, "current_agent_player": "player_2", "opponent": SmartRandomOpponent(player="player_1", center_weight=CENTER_WEIGHT)}
    )
    player_2_model.load_model_for("player_2", '/Users/faizahmed/Documents/SJSU/Reinforcement Learning/pa5/experiments/full_random/checkpoints/player_2/1733113225.488374/actor.pth')

    current_directory = os.path.dirname(os.path.abspath(__file__))

    print("Training against smart random opponent with center weight", CENTER_WEIGHT)

    player_1_model.learn(total_timesteps=-1)
    player_1_model.save_model(path=f"{current_directory}/checkpoints/")

    player_2_model.learn(total_timesteps=-1)
    player_2_model.save_model(path=f"{current_directory}/checkpoints/")
