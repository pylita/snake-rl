import torch
from agent import DeepQLearningAgent

agent = DeepQLearningAgent(
    board_size=10,
    frames=2,
    n_actions=4,
    buffer_size=80000,
    version="v17.1"
)

# Create two dummy Snake states
x = torch.zeros(2, 2, 10, 10, device=agent._device)

with torch.no_grad():
    output = agent._model(x)

print("Device:", agent._device)
print("Output shape:", output.shape)
print("Parameter count:", sum(
    p.numel() for p in agent._model.parameters()
))

# GA02-T01 Part 2: Target-network verification

online = agent._model
target = agent._target_net

assert target is not None, "Target network was not created"

# The networks must be separate objects
assert online is not target

online_params = list(online.parameters())
target_params = list(target.parameters())

assert len(online_params) == len(target_params)

# Initially, their weights should be identical
assert all(
    torch.equal(p1, p2)
    for p1, p2 in zip(online_params, target_params)
)

# But they must not share parameter storage
assert all(
    p1.data_ptr() != p2.data_ptr()
    for p1, p2 in zip(online_params, target_params)
)

# Only the online network should be trainable
assert all(p.requires_grad for p in online_params)
assert not any(p.requires_grad for p in target_params)

# The target network should be in evaluation mode
assert not target.training

# Forward-pass results should be finite
assert torch.isfinite(output).all().item()

with torch.no_grad():
    target_output = target(x)

# Same initial weights and input should give matching predictions
assert torch.allclose(output, target_output)

print("Target-network verification: PASSED")