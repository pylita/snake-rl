
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from agent import DeepQLearningAgent
from game_environment import Snake


# GA02-T10: Live demonstration configuration
VERSION = "v17.1"
#CHECKPOINT = 200000
#MODEL_DIR = "models/v17.1_train_cpu"
MODEL_DIR = "models/v17.1_t12_fresh_cpu"
CHECKPOINT = 225000

N_GAMES = 3
FPS = 10
SEED = 42


with open(f"model_config/{VERSION}.json", "r") as f:
    config = json.load(f)

np.random.seed(SEED)

# Single-game environment for visualization
env = Snake(
    board_size=config["board_size"],
    frames=config["frames"],
    max_time_limit=config["max_time_limit"],
    obstacles=bool(config["obstacles"]),
    version=VERSION,
)

# Create the trained agent
agent = DeepQLearningAgent(
    board_size=config["board_size"],
    frames=config["frames"],
    n_actions=config["n_actions"],
    buffer_size=10,
    version=VERSION,
)

agent.load_model(
    file_path=MODEL_DIR,
    iteration=CHECKPOINT,
)
agent._model.eval()

print(f"Loaded checkpoint: {CHECKPOINT}")
print(f"Device: {agent._device}")
print("Starting live demonstration...")

# Original board value mapping:
# 0 = empty, 1 = body, 2 = head, 3 = food, 4 = obstacle
colors = ListedColormap([
    "lightgray",
    "green",
    "lightgreen",
    "red",
    "dimgray",
])

plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

state = env.reset()
image = ax.imshow(
    state[:, :, 0],
    cmap=colors,
    vmin=-0.5,
    vmax=4.5,
    interpolation="nearest",
)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("GA02 — Trained DQN Snake")

plt.show(block=False)

for game in range(1, N_GAMES + 1):
    if not plt.fignum_exists(fig.number):
        break

    state = env.reset()
    done = False
    info = {"food": 0, "time": 0}

    print(f"\nGame {game}/{N_GAMES}")
    seen_states = {}
    loop_detected = False
    last_food_count = 0

    while not done and plt.fignum_exists(fig.number):
        # The trained DQN chooses the best legal action.
        legal_moves = env.get_legal_moves()

        # Detect repeated states under the greedy policy.
        signature = (state.tobytes(), legal_moves.tobytes())

        if signature in seen_states:
            cycle_length = info["time"] - seen_states[signature]

            print(
                f"Loop detected at step {info['time']} "
                f"(cycle length: {cycle_length} steps)"
            )

            loop_detected = True
            ax.set_title(f"Game {game}/{N_GAMES} | Loop detected")
            plt.pause(1.5)
            break

        seen_states[signature] = info["time"]

        action = agent.move(
            state,
            legal_moves,
            env.get_values(),
        )

        next_state, reward, done, info, _ = env.step(action)

        state = next_state.copy()

        if info["food"] > last_food_count:
            seen_states.clear()
            last_food_count = info["food"]

        # Display the latest board.
        image.set_data(state[:, :, 0])

        snake_length = 2 + info["food"]

        ax.set_title(
            f"Game {game}/{N_GAMES} | "
            f"Length: {snake_length} | "
            f"Steps: {info['time']}"
        )

        plt.pause(1 / FPS)

    reason = (
    "loop_detected"
    if loop_detected
    else info.get("termination_reason")
    )

    print(
        f"Final length: {2 + info['food']} | "
        f"Steps: {info['time']} | "
        f"Reason: {reason}"
    )

    if plt.fignum_exists(fig.number):
        plt.pause(1.0)

print("\nDemonstration finished.")

plt.ioff()
if plt.fignum_exists(fig.number):
    plt.show()
