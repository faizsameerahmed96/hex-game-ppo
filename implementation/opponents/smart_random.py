import random
import numpy as np  # For easier manipulation of probabilities

from opponents.opponent_interface import OpponentInterface


class SmartRandomOpponent(OpponentInterface):
    def __init__(self, player="player_1", center_weight=1.0):
        super().__init__("SmartRandomOpponent")
        self.player = player
        self.center_weight = center_weight  # Parameter to adjust weighting towards center
        self.target_col = random.randint(0, 10)  # Random column target for player_1
        self.target_row = random.randint(0, 10)  # Random row target for player_2

    def reset(self):
        self.target_col = random.randint(0, 10)
        self.target_row = random.randint(0, 10)

    def get_action(self, observation):
        board = observation["observation"]
        board_size = len(board)

        if board_size != 11:  # Ensure this behavior only for 11x11 boards
            raise ValueError("This strategy is designed for an 11x11 board.")

        # Get all empty cells
        zero_indexes = [
            (i, j)
            for i in range(board_size)
            for j in range(board_size)
            if board[i][j] == 0
        ]

        if not zero_indexes:
            raise ValueError("No valid moves available on the board.")

        if self.player == "player_1":
            # Player 1: Column-wise strategy
            probabilities = [
                max(0, 1.0 - abs(self.target_col - col) * 0.1 * self.center_weight)
                for col in range(board_size)
            ]
            action_weights = [
                probabilities[j] for _, j in zero_indexes
            ]

        elif self.player == "player_2":
            # Player 2: Row-wise strategy
            probabilities = [
                max(0, 1.0 - abs(self.target_row - row) * 0.1 * self.center_weight)
                for row in range(board_size)
            ]
            action_weights = [
                probabilities[i] for i, _ in zero_indexes
            ]

        else:
            raise ValueError("Invalid player. Expected 'player_1' or 'player_2'.")

        # Normalize probabilities so they sum to 1
        action_weights = np.array(action_weights)
        if action_weights.sum() == 0:
            # If all weights are zero, distribute equally across all empty cells
            action_weights = np.ones_like(action_weights) / len(action_weights)
        else:
            action_weights = action_weights / action_weights.sum()

        # Choose an action based on the weighted probabilities
        chosen_index = random.choices(range(len(zero_indexes)), weights=action_weights, k=1)[0]
        row, col = zero_indexes[chosen_index]

        # Convert row, col to action
        action = row * board_size + col
        return action
