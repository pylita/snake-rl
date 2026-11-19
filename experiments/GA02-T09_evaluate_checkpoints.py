
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent import DeepQLearningAgent
from game_environment import SnakeNumpy
from utils import play_game2


VERSION = "v17.1"
#CHECKPOINTS = [132500, 199500, 200000]
CHECKPOINTS = [
    (200000, "models/v17.1_train_cpu"),
    (229000, "models/v17.1_continue_cpu"),
    (250000, "models/v17.1_continue_cpu"),
]
CHECKPOINT_DIR = "models/v17.1_train_cpu"

PARALLEL_GAMES = 8
MIN_COMPLETED_GAMES = 200
SEED = 42

#OUTPUT = Path("experiment_results/GA02-T09_checkpoint_eval.csv")
OUTPUT = Path("experiment_results/GA02-T11_checkpoint_eval.csv")

with open(f"model_config/{VERSION}.json", "r") as f:
    config = json.load(f)

agent = DeepQLearningAgent(
    board_size=config["board_size"],
    frames=config["frames"],
    n_actions=config["n_actions"],
    buffer_size=config["buffer_size"],
    version=VERSION,
)

results = []

for iteration, checkpoint_dir in CHECKPOINTS:
    print(f"\nEvaluating checkpoint {iteration} ...")

    # Load the saved PyTorch weights
    agent.load_model(
        file_path=checkpoint_dir,
        iteration=iteration,
    )
    agent._model.eval()

    # Use a reproducible starting seed
    np.random.seed(SEED)

    # Fresh environment for each checkpoint
    env = SnakeNumpy(
        board_size=config["board_size"],
        frames=config["frames"],
        max_time_limit=config["max_time_limit"],
        games=PARALLEL_GAMES,
        frame_mode=True,
        obstacles=bool(config["obstacles"]),
        version=VERSION,
    )

    total_reward, total_length, completed_games = play_game2(
        env,
        agent,
        config["n_actions"],
        n_games=PARALLEL_GAMES,
        epsilon=-1,
        record=False,
        reset_seed=True,
        sample_actions=False,
        frame_mode=True,
        total_games=MIN_COMPLETED_GAMES,
        stateful=False,
    )

    completed_games = int(completed_games)

    result = {
        "iteration": iteration,
        "games": completed_games,
        "reward_mean": float(total_reward) / completed_games,
        "length_mean": float(total_length) / completed_games,
    }

    results.append(result)
    print(result)


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(results)
df.to_csv(OUTPUT, index=False)

print("\nFINAL COMPARISON")
print(df.to_string(index=False))
print(f"\nSaved results to {OUTPUT}")
