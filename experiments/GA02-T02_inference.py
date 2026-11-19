import numpy as np
from agent import DeepQLearningAgent

agent = DeepQLearningAgent(
    board_size=10,
    frames=2,
    n_actions=4,
    buffer_size=80000,
    version="v17.1"
)

# Two synthetic Snake boards
boards = np.zeros((2, 10, 10, 2), dtype=np.float32)
boards[0, 3, 4, 0] = 4.0
boards[1, 6, 7, 1] = 2.0

original = boards.copy()

# Test batch inference
batch_output = agent._get_model_outputs(boards)

print("Device:", agent._device)
print("Input shape:", boards.shape)
print("Output shape:", batch_output.shape)
print("Output type:", type(batch_output))

assert batch_output.shape == (2, 4)
assert isinstance(batch_output, np.ndarray)
assert np.isfinite(batch_output).all()
assert np.array_equal(boards, original)

# Test single-board inference
single_output = agent._get_model_outputs(boards[0])
assert single_output.shape == (1, 4)

# Test target-network inference
target_output = agent._get_model_outputs(
    boards, model=agent._target_net
)
assert np.allclose(batch_output, target_output, atol=1e-5)

# Check normalization separately
normalized = agent._prepare_input(boards)
assert normalized[0, 3, 4, 0] == 1.0
assert normalized[1, 6, 7, 1] == 0.5

print("GA02-T02: All inference checks PASSED")