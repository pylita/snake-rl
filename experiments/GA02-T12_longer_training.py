"""GA02-T12: launch an uninterrupted 400k-iteration Snake DQN training run.

Place this file in snake-rl/experiments/ and run from the repository root:
    python -m experiments.GA02-T12_longer_training

Requires training.py to support GA02_T12=1, episodes=400_000, and the unique
run tag `v17.1_t12_fresh_cpu`. See the accompanying chat instructions.

This is a launcher and result collector; it intentionally does not duplicate
or alter the learning algorithm in training.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
VERSION = "v17.1"
RUN_TAG = "v17.1_t12_fresh_cpu"
ITERATIONS = 400_000
LOG_INTERVAL = 500
SEED = 42

TRAINING_SCRIPT = ROOT / "training.py"
CONFIG_PATH = ROOT / "model_config" / f"{VERSION}.json"
TRAINING_LOG = ROOT / "model_logs" / f"{RUN_TAG}.csv"
MODEL_DIR = ROOT / "models" / RUN_TAG
RESULT_DIR = ROOT / "experiment_results"
MANIFEST = RESULT_DIR / "GA02-T12_run_manifest.json"
WINDOW_CSV = RESULT_DIR / "GA02-T12_iterations_200000_to_250000.csv"
MILESTONES_CSV = RESULT_DIR / "GA02-T12_milestones.csv"
SUMMARY_JSON = RESULT_DIR / "GA02-T12_run_summary.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if not TRAINING_SCRIPT.is_file():
        raise SystemExit(f"Cannot find training.py: {TRAINING_SCRIPT}")
    if not CONFIG_PATH.is_file():
        raise SystemExit(f"Cannot find configuration: {CONFIG_PATH}")

    training_text = TRAINING_SCRIPT.read_text(encoding="utf-8")
    if "GA02_T12" not in training_text or "T12_TEST" not in training_text:
        raise SystemExit(
            "T12 mode has not been added to training.py. "
            "Follow the three training.py edits in the chat first."
        )

    if TRAINING_LOG.exists() or MODEL_DIR.exists() or MANIFEST.exists():
        raise SystemExit(
            "T12 output already exists. Stopping to avoid overwriting an "
            "existing experiment. Inspect model_logs/, models/, and "
            "experiment_results/ before starting another run."
        )

    # Make the device requirement explicit before the overnight run begins.
    import torch
    if (hasattr(torch, "xpu") and torch.xpu.is_available()) or torch.cuda.is_available():
        raise SystemExit("T12 is configured for CPU. Activate the uitnn environment first.")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if bool(config.get("supervised", False)):
        raise SystemExit("T12 must start from scratch: supervised must be false.")

    # Explicitly disable all earlier modes, even if the user left their
    # Windows CMD environment variables enabled in this terminal.
    env = os.environ.copy()
    env.update(
        GA02_SMOKE="0",
        GA02_BENCH="0",
        GA02_CONTINUE="0",
        GA02_T12="1",
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "GA02-T12",
        "design": "Fresh, uninterrupted DQN training; do not load existing checkpoints",
        "start_local": now_iso(),
        "run_tag": RUN_TAG,
        "target_iterations": ITERATIONS,
        "logging_and_target_update_every": LOG_INTERVAL,
        "seed_expected_in_training_py": SEED,
        "python_executable": sys.executable,
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "config": config,
        "training_log": str(TRAINING_LOG.relative_to(ROOT)),
        "checkpoint_dir": str(MODEL_DIR.relative_to(ROOT)),
        "window_csv": str(WINDOW_CSV.relative_to(ROOT)),
        "milestones_csv": str(MILESTONES_CSV.relative_to(ROOT)),
        "status": "started",
    }
    save_json(MANIFEST, manifest)

    print("\nGA02-T12 — fresh uninterrupted training", flush=True)
    print(f"Target: {ITERATIONS:,} iterations | seed: {SEED} | CPU", flush=True)
    print(f"Run tag: {RUN_TAG}", flush=True)
    print("Verify the next line says: Mode: T12, device: cpu, iterations: 400000", flush=True)
    print("The launcher will not load the T08 or T11 checkpoint.\n", flush=True)

    # Execute the unchanged training loop in a fresh Python process.
    # Inherit stdout/stderr so tqdm progress remains visible in the terminal.
    command = [sys.executable, "-u", "training.py"]
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    manifest["end_local"] = now_iso()
    manifest["exit_code"] = completed.returncode
    manifest["status"] = "completed" if completed.returncode == 0 else "failed"
    save_json(MANIFEST, manifest)

    if TRAINING_LOG.is_file():
        import pandas as pd

        df = pd.read_csv(TRAINING_LOG)
        expected_columns = {"iteration", "reward_mean", "length_mean", "games", "loss"}
        if not expected_columns.issubset(df.columns):
            raise SystemExit(f"Unexpected training log columns: {list(df.columns)}")
        df = df.sort_values("iteration").copy()

        # Moving averages use 10 logged evaluations = 5,000 iterations.
        df["length_mean_ma10"] = df["length_mean"].rolling(10, min_periods=1).mean()
        df["reward_mean_ma10"] = df["reward_mean"].rolling(10, min_periods=1).mean()

        window = df.loc[df["iteration"].between(200_000, 250_000)]
        window.to_csv(WINDOW_CSV, index=False)

        milestones = [
            500, 10_000, 50_000, 100_000, 150_000,
            200_000, 205_000, 210_000, 215_000, 220_000,
            225_000, 229_000, 230_000, 235_000, 240_000,
            245_000, 250_000, 275_000, 300_000, 350_000, 400_000,
        ]
        snapshot = df.loc[df["iteration"].isin(milestones)]
        snapshot.to_csv(MILESTONES_CSV, index=False)

        best_idx = df["length_mean"].idxmax()
        summary = {
            "experiment": "GA02-T12",
            "run_status": manifest["status"],
            "last_logged_iteration": int(df["iteration"].iloc[-1]),
            "total_log_records": int(len(df)),
            "window_200k_250k_records": int(len(window)),
            "best_small_sample_mean_length": {
                "iteration": int(df.loc[best_idx, "iteration"]),
                "length_mean": float(df.loc[best_idx, "length_mean"]),
                "games": int(df.loc[best_idx, "games"]),
            },
            "note": "Training-time evaluations are only ~8 games; run independent "
                    "200+ game tests on selected checkpoints before comparing policies.",
        }
        save_json(SUMMARY_JSON, summary)
        print("\nT12 results saved:", flush=True)
        print(f"  {TRAINING_LOG.relative_to(ROOT)}", flush=True)
        print(f"  {WINDOW_CSV.relative_to(ROOT)}", flush=True)
        print(f"  {MILESTONES_CSV.relative_to(ROOT)}", flush=True)
        print(f"  {SUMMARY_JSON.relative_to(ROOT)}", flush=True)
        print(f"  Checkpoint folder: {MODEL_DIR.relative_to(ROOT)}", flush=True)
    else:
        print("Training log not created. Inspect the traceback above.", flush=True)

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    if not (MODEL_DIR / "model_400000.pt").is_file():
        raise SystemExit("Training ended but final checkpoint model_400000.pt is missing.")
    print("\nGA02-T12 successfully finished all 400,000 iterations.", flush=True)


if __name__ == "__main__":
    main()
