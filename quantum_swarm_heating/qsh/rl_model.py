import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

class SimpleQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x):
        return self.fc(x)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = SimpleQNet(state_dim, action_dim)
        self.critic = SimpleQNet(state_dim, 1)

    def forward(self, x):
        action = self.actor(x)
        value = self.critic(x)
        return action, value

def train_rl(model, optimizer, episodes=20):
    state_ranges = torch.tensor([0.5, 100, 5, 55, 15, 5, 20, 10, 5, 10, 5, 2, 2, 5, 1.0])
    for _ in range(episodes):
        sim_state = torch.rand(15) * state_ranges
        action, value = model(sim_state.unsqueeze(0))
        action = action.squeeze(0)

        sim_demand = sim_state[4].item()
        sim_mode = 1.0 if sim_demand > 5 else 0.0
        sim_flow = 45.0 if sim_demand > 5 else 35.0
        norm_flow = (sim_flow - 30) / 20

        mode_match = (torch.sigmoid(action[0]) > 0.5) == (sim_mode == 1.0)
        flow_err = abs(action[1].item() - norm_flow)
        sim_reward = 1.0 if mode_match and flow_err < 0.1 else -1.0

        td_error = sim_reward - value.item()
        critic_loss = td_error ** 2

        mode_target = torch.tensor(sim_mode, dtype=torch.float32)
        mode_log_prob = F.binary_cross_entropy_with_logits(action[0], mode_target)
        flow_target = torch.tensor(norm_flow, dtype=torch.float32)
        flow_mse = (action[1] - flow_target).pow(2)
        actor_loss = mode_log_prob + flow_mse

        loss = critic_loss + 0.5 * actor_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info("Initial RL training complete with simulated scenarios.")
