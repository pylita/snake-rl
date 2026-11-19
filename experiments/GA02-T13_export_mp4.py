"""GA02-T13: export a greedy trained Snake policy to an MP4 video.

Copy this standalone file to snake-rl/experiments/GA02-T13_export_mp4.py.
Run from the snake-rl repository root:

    python -m experiments.GA02-T13_export_mp4

Optional video-only dependency:
    python -m pip install imageio-ffmpeg

The model weights and the environment are NOT modified. Gameplay ends when the
Snake terminates, a repeated state is detected, or --max-steps is reached.
These truncated demonstrations must not be treated as independent evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import imageio_ffmpeg
except ImportError as exc:
    raise SystemExit(
        "Video encoding requires imageio-ffmpeg. In the active conda environment, "
        "run: python -m pip install imageio-ffmpeg"
    ) from exc

# The existing game environment reads model_config/ and models/ relative to cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

from agent import DeepQLearningAgent  # noqa: E402
from game_environment import Snake   # noqa: E402

VERSION = "v17.1"
IMAGE_SIZE = (800, 960)  # Both divisible by 16 for reliable H.264 encoding.
BOARD_START = (50, 140)
CELL_SIZE = 70
COLORS = {
    0: (238, 242, 246),   # empty
    1: (24, 137, 76),     # body
    2: (126, 230, 143),   # head
    3: (245, 87, 66),     # food
    4: (81, 89, 101),     # wall / obstacle
}


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = font(33)
FONT_BODY = font(24)
FONT_SMALL = font(18)


def render_frame(board: np.ndarray, game: int, total_games: int,
                 length: int, steps: int, iteration: int,
                 message: str = "Greedy policy (no exploration)") -> bytes:
    """Render a full video frame as RGB24 bytes for FFmpeg."""
    frame = Image.new("RGB", IMAGE_SIZE, (17, 25, 38))
    d = ImageDraw.Draw(frame)
    d.text((50, 27), "GA02  |  PYTORCH DQN SNAKE", font=FONT_TITLE, fill=(238, 245, 252))
    d.text((51, 84), f"Game {game}/{total_games}    Length {length}    Steps {steps}",
           font=FONT_BODY, fill=(180, 217, 245))

    # Board: state frame 0 is the newest frame in the original environment.
    current = board[:, :, 0]
    for row in range(current.shape[0]):
        for col in range(current.shape[1]):
            x = BOARD_START[0] + col * CELL_SIZE
            y = BOARD_START[1] + row * CELL_SIZE
            d.rectangle((x + 1, y + 1, x + CELL_SIZE - 2, y + CELL_SIZE - 2),
                        fill=COLORS.get(int(current[row, col]), (245, 213, 90)))

    d.text((50, 865), f"Model v17.1  |  checkpoint {iteration:,}",
           font=FONT_SMALL, fill=(211, 220, 232))
    d.text((50, 900), message[:65], font=FONT_SMALL, fill=(161, 194, 217))
    return frame.tobytes()


def run(args: argparse.Namespace) -> None:
    with open(PROJECT_ROOT / "model_config" / "v17.1.json", encoding="utf-8") as f:
        config = json.load(f)

    model_dir = Path(args.model_dir)
    model_path = model_dir / f"model_{args.iteration:04d}.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {model_path}")

    obstacle_path = PROJECT_ROOT / "models" / VERSION / "obstacles_board"
    if config["obstacles"] and not obstacle_path.is_file():
        raise FileNotFoundError(f"Required v17.1 obstacle data is missing: {obstacle_path}")

    np.random.seed(args.seed)
    agent = DeepQLearningAgent(
        board_size=config["board_size"],
        frames=config["frames"],
        n_actions=config["n_actions"],
        buffer_size=10,  # Not used for inference.
        version=VERSION,
    )
    agent.load_model(file_path=str(model_dir), iteration=args.iteration)
    agent._model.eval()

    env = Snake(
        board_size=config["board_size"],
        frames=config["frames"],
        max_time_limit=config["max_time_limit"],
        obstacles=bool(config["obstacles"]),
        version=VERSION,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio_ffmpeg.write_frames(
        str(output.resolve()), size=IMAGE_SIZE, fps=args.fps,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        output_params=["-crf", "22", "-movflags", "+faststart"],
    )
    writer.send(None)
    completed = 0
    try:
        for game in range(1, args.games + 1):
            state = env.reset()
            info = {"food": 0, "time": 0, "termination_reason": None}
            seen = {}
            last_food = 0
            stop_reason = "max_steps"

            for _ in range(args.max_steps):
                legal = env.get_legal_moves()
                # Detection only; no action is altered by the demonstration.
                signature = (state.tobytes(), legal.tobytes())
                if signature in seen:
                    stop_reason = "repeated state / loop"
                    break
                seen[signature] = int(info["time"])

                action = agent.move(state, legal, env.get_values())
                next_state, reward, done, info, _ = env.step(action)
                state = next_state.copy()

                if info["food"] > last_food:
                    last_food = int(info["food"])
                    seen.clear()

                writer.send(render_frame(
                    state, game, args.games, 2 + int(info["food"]),
                    int(info["time"]), args.iteration,
                ))

                if done:
                    stop_reason = str(info.get("termination_reason") or "game ended")
                    break

            final_frame = render_frame(
                state, game, args.games, 2 + int(info["food"]),
                int(info["time"]), args.iteration,
                message=f"Game ended: {stop_reason}",
            )
            # Keep the final frame visible briefly, including loop termination.
            for _ in range(max(1, args.fps)):
                writer.send(final_frame)
            completed += 1
            print(
                f"Game {game}/{args.games}: length={2 + int(info['food'])}, "
                f"steps={info['time']}, reason={stop_reason}", flush=True
            )
    finally:
        writer.close()

    print(f"Saved MP4: {output.resolve()} ({completed} games, {args.fps} fps)")
    print("Note: loop detection and max-step limits are for video demonstration, not scoring.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained Snake policy as MP4")
    parser.add_argument("--model-dir", default="models/v17.1_t12_fresh_cpu")
    parser.add_argument("--iteration", type=int, default=225000)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="videos/GA02_snake_policy_v17.1_225000.mp4")
    args = parser.parse_args()
    if args.games < 1 or args.fps < 1 or args.max_steps < 1:
        parser.error("--games, --fps and --max-steps must all be positive")
    run(args)


if __name__ == "__main__":
    main()
