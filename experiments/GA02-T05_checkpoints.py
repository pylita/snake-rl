
import copy
import tempfile
from pathlib import Path

import numpy as np
import torch
from agent import DeepQLearningAgent

rng = np.random.default_rng(42)
torch.manual_seed(42)

agent = DeepQLearningAgent(
    board_size=10,
    frames=2,
    n_actions=4,
    buffer_size=80000,
    version="v17.1"
)

# 1. Generate synthetic experiences
n = 64

s = rng.integers(0, 5, (n, 10, 10, 2), dtype=np.uint8)
next_s = rng.integers(0, 5, (n, 10, 10, 2), dtype=np.uint8)
actions = rng.integers(0, 4, n, dtype=np.uint8)
rewards = rng.choice([-1, 0, 1], n).astype(np.int16)

done = np.zeros(n, dtype=np.uint8)
done[::5] = 1

legal_moves = np.ones((n, 4), dtype=np.uint8)
legal_moves[::5] = 0

agent.add_to_buffer(
    s, actions, rewards, next_s, done, legal_moves
)

# 2. Train once so RMSprop has optimizer state
loss = agent.train_agent(batch_size=64, reward_clip=True)
assert np.isfinite(loss)
assert len(agent._optimizer.state) > 0

# Save copies of the original values for comparison
online_before = [
    p.detach().clone() for p in agent._model.parameters()
]
target_before = [
    p.detach().clone() for p in agent._target_net.parameters()
]
optimizer_before = copy.deepcopy(agent._optimizer.state_dict())

with tempfile.TemporaryDirectory() as tmpdir:

    # 3. Save checkpoint
    agent.save_model(file_path=tmpdir, iteration=100)

    checkpoint_path = Path(tmpdir) / "model_0100.pt"
    assert checkpoint_path.is_file()
    print("Test 1: Checkpoint saving PASSED")

    # 4. Deliberately change both networks
    with torch.no_grad():
        next(agent._model.parameters()).add_(1.0)
        next(agent._target_net.parameters()).add_(1.0)

    # 5. Restore checkpoint
    agent.load_model(file_path=tmpdir, iteration=100)

    assert all(
        torch.equal(old, new)
        for old, new in zip(online_before, agent._model.parameters())
    )

    assert all(
        torch.equal(old, new)
        for old, new in zip(target_before, agent._target_net.parameters())
    )

    print("Test 2: Network weight restoration PASSED")

    # 6. Verify optimizer state restoration
    optimizer_after = agent._optimizer.state_dict()

    assert optimizer_before["param_groups"] == optimizer_after["param_groups"]
    assert optimizer_before["state"].keys() == optimizer_after["state"].keys()

    for key, state in optimizer_before["state"].items():
        for name, value in state.items():
            restored = optimizer_after["state"][key][name]

            if torch.is_tensor(value):
                assert torch.equal(
                    value.detach().cpu(),
                    restored.detach().cpu()
                )
            else:
                assert value == restored

    print("Test 3: RMSprop state restoration PASSED")

    # 7. Verify a fresh agent can load the checkpoint
    restored_agent = DeepQLearningAgent(
        board_size=10,
        frames=2,
        n_actions=4,
        buffer_size=80000,
        version="v17.1"
    )

    restored_agent.load_model(file_path=tmpdir, iteration=100)

    assert all(
        torch.equal(old, new)
        for old, new in zip(
            online_before, restored_agent._model.parameters()
        )
    )

    print("Test 4: Fresh agent loading PASSED")

print("Device:", agent._device)
print("GA02-T05: ALL TESTS PASSED")
