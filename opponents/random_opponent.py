import random

from opponents.opponent_interface import OpponentInterface


class RandomOpponent(OpponentInterface):
    def __init__(self):
        super().__init__("RandomOpponent")

    def get_action(self, observation):
        zero_indexes = [
            (i, j)
            for i in range(len(observation["observation"]))
            for j in range(len(observation["observation"][i]))
            if observation["observation"][i][j] == 0
        ]
        row, col = random.choice(zero_indexes)
        board_size = len(observation["observation"])
        action = row * board_size + col
        return action

    def reset(self):
        pass