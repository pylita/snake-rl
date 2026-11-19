# DTE2502 — Graded Assignment 02: Snake Deep Q-Learning in PyTorch

This project ports the **DeepQLearningAgent** from the original TensorFlow/Keras [Snake reinforcement-learning repository](https://github.com/DragonWarrior15/snake-rl) to **PyTorch**, using the **v17.1** obstacle-board configuration. The agent receives two consecutive frames of a 10 × 10 Snake board and predicts Q-values for four movement actions.

**Student repository:** https://github.com/pylita/snake-rl  
**Development branch:** `ga02-pytorch` (verify that this branch has been pushed before submission).

## Running Graded assignment 02

These instructions are intended to be run **from the `snake-rl` directory**, which contains `agent.py`, `training.py`, and `model_config/`.

### 1. Environment and dependencies

The implementation has been run in the course-compatible **`uitnn`** conda environment (PyTorch CPU). A separate `torch_xpu` environment was used for optional Intel GPU experiments, but it is **not required** to evaluate the submission.

In **Windows CMD**:

```cmd
conda activate uitnn
python -c "import torch, numpy, pandas, matplotlib, tqdm; print('Core dependencies OK'); print('PyTorch:', torch.__version__)"
```

Core Python packages used by the project are `torch`, `numpy`, `pandas`, `matplotlib`, and `tqdm`. If your environment lacks them, install the missing packages using your course's supported PyTorch installation and the accompanying **`requirements_GA02.txt`**. No TensorFlow or Keras installation is required for the DQN implementation.

**Important:** The repository's original `requirements.txt` was written for the legacy TensorFlow project; it includes obsolete Windows/Python 3.7 wheels and unrelated URLs. **Do not install from that file unchanged.** Replace it with `requirements_GA02.txt` (or a verified equivalent) before distributing setup instructions.

### 2. Required files

For the supplied trained v17.1 demonstration, verify that these files exist:

```text
snake-rl/
├── agent.py
├── training.py
├── game_environment.py
├── replay_buffer.py
├── utils.py
├── model_config/
│   └── v17.1.json
├── models/
│   ├── v17.1/
│   │   └── obstacles_board
│   └── v17.1_t12_fresh_cpu/
│       └── model_225000.pt
└── experiments/
    ├── GA02-T10_live_demo.py
    └── GA02-T13_export_mp4.py
```

The `obstacles_board` data file is needed because v17.1 uses obstacle boards. **Do not delete it.**

**Selected checkpoint:** `models/v17.1_t12_fresh_cpu/model_225000.pt` (~1.48 MB). It is present in the reviewed GA02 archive. If you distribute source code without model binaries, place the model in the same path after downloading it from your Google Drive submission link:

**Model download:** https://drive.google.com/drive/folders/1I6tpKKeoL3Tz7HuUxwd2SZsaI2d7QpXe

### 3. Show the trained policy live

```cmd
python -m experiments.GA02-T10_live_demo
```

This loads the **225,000-iteration** PyTorch checkpoint and opens a Matplotlib window in which the agent chooses actions greedily (no exploratory random actions). The demonstration shows the board, game length, and step counter. Its loop detector can end a repetitive game early and begin another game **without changing the model's actions or weights**.

Close the figure window to stop the live demonstration.

### 4. Export an MP4 gameplay video

Copy the separate file **`GA02-T13_export_mp4.py`** into the `experiments/` directory. The core DQN does not need any extra dependency, but **video export requires** the optional FFmpeg encoder package:

```cmd
python -m pip install imageio-ffmpeg
```

Then, from the repository root:

```cmd
python -m experiments.GA02-T13_export_mp4
```

The default command generates:

```text
videos/GA02_snake_policy_v17.1_225000.mp4
```

Defaults: selected 225k checkpoint, seed 42, **2 games**, **12 frames/s**, up to **350 steps per game**. The encoder writes an H.264 MP4, and the script prints each game's final length and termination reason.

To record a longer or shorter clip:

```cmd
python -m experiments.GA02-T13_export_mp4 --games 3 --fps 12 --max-steps 500 --output videos/snake_demo.mp4
```

To export the original 200k checkpoint instead:

```cmd
python -m experiments.GA02-T13_export_mp4 --model-dir models/v17.1_train_cpu --iteration 200000 --output videos/snake_200k.mp4
```

**Video interpretation:** The exporter does not train or override the agent's actions. It can stop a game on a repeated state or after the specified step limit to keep videos usable. Thus, video-demo game lengths should **not** be treated as unbiased, independent performance evaluations.

**Gameplay video:** _[Add the verified video URL here after uploading the MP4, if sharing externally.]_

### 5. Evaluate the trained policy

The original evaluation uses `SnakeNumpy` and `play_game2()` with exploration disabled and `record=False`. The independent 200-game comparisons already generated during development are available at:

```text
experiment_results/GA02-T09_checkpoint_eval.csv
experiment_results/GA02-T11_checkpoint_eval.csv
experiment_results/GA02-T12_checkpoint_eval.csv
```

For the selected T12 checkpoint at 225k, the **200-game independent mean snake length was 22.270** and the mean reward was **19.465**. The original 200k baseline measured **20.585** mean length. These are empirical averages, not guarantees for every game.

Note that `GA02-T12_checkpoint_eval.py` currently lists some checkpoints that were **not retained in the cleaned archive** (including T12 233k, 250k, and 301k). To rerun the complete nine-checkpoint table, restore those checkpoints or adjust its list to the models available. The saved CSV is the record of the earlier full comparison.

### 6. Train from scratch (optional)

```cmd
set GA02_SMOKE=0
set GA02_BENCH=0
set GA02_CONTINUE=0
set GA02_T12=0
python training.py
```

The default mode performs **200,000 training iterations** using v17.1 and saves evaluation logs and checkpoints every 500 iterations. For a quick end-to-end test:

```cmd
set GA02_SMOKE=1
python training.py
```

**Caution:** A new full training run writes to `models/v17.1_train_cpu/` and `model_logs/v17.1_train_cpu.csv`; copy any results you want to preserve before starting a new run in the same folder. The smoke mode has a separate output name. A fresh 400k experiment was conducted using the separate `GA02-T12_longer_training.py` launcher.

## Implementation summary

The v17.1 model contains three convolutions with output channels **16 → 32 → 64**, followed by Flatten, a 64-neuron hidden layer, and four linear Q-value outputs. The first 3×3 convolution uses same padding; the next 3×3 and 5×5 convolutions use valid padding. The network has **122,068 trainable parameters**.

The PyTorch DQN implementation provides:

- Channel-order conversion: TensorFlow/Keras NHWC to PyTorch NCHW (`permute(0,3,1,2)`).
- Online and target networks; target weights synchronized every **500 iterations**.
- Legal-action masking; epsilon-greedy exploration during training.
- Replay buffer capacity **80,000**, training batch size **64**.
- Discount factor **0.99**, RMSprop learning rate **0.0005**, and Huber loss (`nn.SmoothL1Loss(beta=1.0)`).
- Terminal-state-safe Bellman targets; PyTorch forward/backward optimization.
- `.pt` checkpoints with online/target model and optimizer state, and checkpoint loading with `map_location="cpu"`.

**Resume limitation:** The `.pt` checkpoint does **not** include the replay buffer, epsilon schedule, environment state, or random-number-generator states. Loading weights and optimizer state is therefore not an exact restoration of the whole training process.

## Experiments and results

| Experiment | Description | Finding |
| --- | --- | --- |
| T01–T05 | Shapes, inference, action masking, gradients, checkpoint reload | CPU and XPU tests passed |
| T06 | Real Snake end-to-end integration | 10 training iterations completed on CPU and XPU |
| T07 | 500-iteration device comparison | CPU **92.69 it/s**; Intel XPU **77.50 it/s** |
| T08 | 200k fresh training | Selected 200k model averaged **20.585** length over 200 games |
| T09 | Independent model selection | Larger evaluations changed the preferred checkpoint |
| T10 | Live policy demonstration | Greedy gameplay and repeat-state loop detection |
| T11 | Continue 200k → 250k with rebuilt replay buffer | At 250k: mean length **18.450** |
| T12 | Fresh uninterrupted 400k training | At 250k: **18.485**; best tested 225k checkpoint: **22.270** |

One training iteration is a cycle of collecting experience and performing a DQN update; it is **not** one complete Snake game. The model's performance varied across checkpoints. Continuing training or preserving the replay buffer did not guarantee improving the average length.

## Project structure

```text
agent.py                  # PyTorch DQN agent; original unused agent classes also remain
training.py               # DQN training loop and experiment modes
game_environment.py       # Snake and vectorized SnakeNumpy environments
replay_buffer.py          # Transition storage / sampling
utils.py                  # Training/evaluation helpers
model_config/v17.1.json   # CNN and game configuration
models/v17.1/obstacles_board            # Required obstacle data
models/v17.1_t12_fresh_cpu/model_225000.pt # Selected trained model
experiments/GA02-T10_live_demo.py       # Live visualization
experiments/GA02-T13_export_mp4.py      # MP4 video exporter (copy separately)
model_logs/               # Training traces
experiment_results/       # Saved checkpoint comparison CSVs
```

## Additional dependencies

- **No TensorFlow/Keras** is needed to run this converted DQN.
- For the **optional MP4 export only**, install **`imageio-ffmpeg`** with `python -m pip install imageio-ffmpeg`. It supplies an FFmpeg encoder for video export without requiring a separate system-wide FFmpeg installation.
- The original core project also uses NumPy, pandas, Matplotlib, and tqdm. `Pillow` is used for video frame drawing and is normally installed as a Matplotlib dependency. If absent, install `Pillow`.

## Attribution and limitations

Based on the original [DragonWarrior15/snake-rl](https://github.com/DragonWarrior15/snake-rl) project. This assignment focuses on converting its `DeepQLearningAgent` for **v17.1** to PyTorch while retaining the game's environment and training methodology where possible. Other original algorithm classes may still contain TensorFlow-specific historical code, but the **DeepQLearningAgent** workflow uses PyTorch.

Known limitations include repetitive action loops on some obstacle boards, fluctuations in DQN performance across training checkpoints, and incomplete checkpoint state for exact training resumption. The live and video demonstrations use a diagnostic loop detector; the reported 200-game evaluation figures come from the separate batch evaluation workflow.
