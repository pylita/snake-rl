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

n = 64
s = rng.integers(0, 5, (n, 10, 10, 2), dtype=np.uint8)
next_s = rng.integers(0, 5, (n, 10, 10, 2), dtype=np.uint8)
actions = rng.integers(0, 4, n, dtype=np.uint8)
rewards = rng.choice([-1, 0, 1], size=n).astype(np.int16)

done = np.zeros(n, dtype=np.uint8)
done[::5] = 1

legal_moves = np.ones((n, 4), dtype=np.uint8)
legal_moves[::5] = 0

agent.add_to_buffer(
    s, actions, rewards, next_s, done, legal_moves
)

online_before = [
    p.detach().clone() for p in agent._model.parameters()
]
target_before = [
    p.detach().clone() for p in agent._target_net.parameters()
]

loss = agent.train_agent(batch_size=64, reward_clip=True)

online_changed = any(
    not torch.equal(old, new)
    for old, new in zip(online_before, agent._model.parameters())
)
target_unchanged = all(
    torch.equal(old, new)
    for old, new in zip(target_before, agent._target_net.parameters())
)

assert np.isfinite(loss)
assert online_changed
assert target_unchanged

print("Device:", agent._device)
print("Loss:", loss)
print("Online weights changed:", online_changed)
print("Target weights unchanged:", target_unchanged)
print("GA02-T04: Training smoke test PASSED")