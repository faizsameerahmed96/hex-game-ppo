import os

def get_latest_model_for_player(player):
    """
    Given a player, returns the latest model for that player from checkpoints.
    """
    path = f"./checkpoints/{player}/actor"
    files = os.listdir(path)
    files.sort()
    return path + "/" + files[-1]