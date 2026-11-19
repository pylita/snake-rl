
import numpy as np
import torch
from agent import DeepQLearningAgent

torch.manual_seed(42)

agent = DeepQLearningAgent(
    board_size=10,
    frames=2,
    n_actions=4,
    buffer_size=80000,
    version="v17.1"
)

print("Device:", agent._device)

# =====================================
# Test 1: Legal action selection
# =====================================

# Predetermined Q-values for two Snake states
fake_q_values = np.array([
    [0.2, 0.8, 5.0, 0.5],
    [10.0, 0.4, 1.2, 9.0]
], dtype=np.float32)

# 1 = legal, 0 = illegal
legal_moves = np.array([
    [1, 1, 0, 1],
    [0, 1, 1, 0]
])

boards = np.zeros((2, 10, 10, 2), dtype=np.float32)

# Temporarily replace the prediction method
# so we can test move() with known Q-values.
original_method = agent._get_model_outputs

try:
    agent._get_model_outputs = lambda board, model=None: fake_q_values

    actions = agent.move(boards, legal_moves)

    assert np.array_equal(actions, [1, 2])
    print("Test 1: Legal action selection PASSED")

finally:
    agent._get_model_outputs = original_method


# =====================================
# Test 2: Target-network synchronization
# =====================================

online = agent._model
target = agent._target_net

assert target is not None

# Initial weights must match
assert all(
    torch.equal(p1, p2)
    for p1, p2 in zip(online.parameters(), target.parameters())
)

# Save the original target parameter
target_before = next(target.parameters()).detach().clone()

# Deliberately change an online parameter
with torch.no_grad():
    next(online.parameters()).add_(0.5)

# Target must remain unchanged
assert torch.equal(
    next(target.parameters()), target_before
)

assert not torch.equal(
    next(online.parameters()), next(target.parameters())
)

print("Test 2: Target independence PASSED")

# Synchronize weights
agent.update_target_net()

# All parameters must match again
assert all(
    torch.equal(p1, p2)
    for p1, p2 in zip(online.parameters(), target.parameters())
)

# But the parameter storage must remain independent
assert all(
    p1.data_ptr() != p2.data_ptr()
    for p1, p2 in zip(online.parameters(), target.parameters())
)

assert not any(p.requires_grad for p in target.parameters())
assert not target.training

print("Test 3: Target synchronization PASSED")
print("GA02-T03: ALL TESTS PASSED")
