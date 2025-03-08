import random
import numpy as np
import torch  # For easier manipulation of probabilities

from opponents.opponent_interface import OpponentInterface
import torch.nn.functional as F


class DuelOpponent(OpponentInterface):
    def __init__(self, player="player_1", actor=None, env=None):
        super().__init__("DuelOpponent")
        self.player = player
        self.actor = actor
        self.temperature = 0.5
        self.env = env

    def reset(self):
        pass

    def get_action(self, observation):
        """
        Queries an action from the actor network, should be called from rollout.

        Parameters:
                obs - the observation at the current timestep

        Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        logits = self.actor(
            torch.tensor(
                np.expand_dims(self.transform_observation(observation).numpy(), axis=0)
            )
        )

        logits = logits / self.temperature

        action_probs = F.softmax(logits, dim=-1)

        action_mask = torch.tensor(
            self.env.generate_info(self.player)["action_mask"]
        )

        action_probs = action_probs * action_mask

        action_probs = action_probs / action_probs.sum()

        action_probs = torch.tensor(action_probs)

        # Sample an action
        action = torch.multinomial(action_probs, num_samples=1).item()

        return action
    
    def transform_observation(self, obs):
        """
        Convert observation into a one hot encoded tensor for conv2d layers

        Parameters:
            obs - the observation dictionary

        Returns:
            Tensor - a flattened tensor representation of the observation
        """
        board = obs["observation"]
        one_hot_board = np.eye(3)[board]  # one hot encode the board
        one_hot_board = np.transpose(
            one_hot_board, (2, 0, 1)
        )  # transpose to (channels, height, width)
        return torch.tensor(one_hot_board, dtype=torch.float32)