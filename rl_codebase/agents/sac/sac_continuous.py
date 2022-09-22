from rl_codebase.core import (
    get_observation_space,
    get_action_space
)
import torch
import gym
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from rl_codebase.agents.sac.models import ContinuousSACActor, Critic


class ContinuousSAC(nn.Module):
    def __init__(
            self,
            observation_space: gym.spaces,
            action_space: gym.spaces,
            learning_rate: float = 3e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            num_layers=3,
            hidden_dim=256,
            init_temperature=.2,
            device='cpu',
            **kwargs,
    ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = ContinuousSACActor(observation_space, action_space, num_layers,
                                        hidden_dim).to(device)

        self.critic = Critic(observation_space, action_space, num_layers, hidden_dim).to(device)

        # Target entropy from the paper
        self.target_entropy = -np.prod(action_space.shape)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic._online_q.parameters(), lr=learning_rate,
        )

        self.log_ent_coef = torch.log(init_temperature *
                                      torch.ones(1, device=device)).requires_grad_(True)

        self.ent_coef_optimizer = torch.optim.Adam(
            [self.log_ent_coef], lr=learning_rate
        )

    def critic_loss(self, batch, log_ent_coef):
        # Compute target Q 
        with torch.no_grad():
            next_pi, next_log_pi = self.actor.sample(batch.next_states, compute_log_pi=True)
            next_q_vals = self.critic.target_q(batch.next_states, next_pi)
            next_q_val = torch.minimum(*next_q_vals)

            ent_coef = torch.exp(log_ent_coef)
            next_q_val = next_q_val - ent_coef * next_log_pi

            target_q_val = batch.rewards + (1 - batch.dones) * self.gamma * next_q_val

        current_q_vals = self.critic.online_q(batch.states, batch.actions)
        critic_loss = .5 * sum(F.mse_loss(current_q, target_q_val) for current_q in current_q_vals)

        return critic_loss

    def actor_loss(self, batch, log_ent_coef):
        pi, log_pi = self.actor.sample(batch.states, compute_log_pi=True)

        q_vals = self.critic.online_q(batch.states, pi)
        q_val = torch.minimum(*q_vals)

        with torch.no_grad():
            ent_coef = torch.exp(log_ent_coef)

        actor_loss = (ent_coef * log_pi - q_val).mean()

        return actor_loss

    def alpha_loss(self, batch, log_ent_coef):
        with torch.no_grad():
            pi, log_pi = self.actor.sample(batch.states, compute_log_pi=True)
        alpha_loss = -(log_ent_coef * (log_pi + self.target_entropy).detach()).mean()

        return alpha_loss

    def update(self, batch):
        # Update critic
        critic_loss = self.critic_loss(batch, self.log_ent_coef)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.polyak_update(self.tau)

        # Update actor
        actor_loss = self.actor_loss(batch, self.log_ent_coef)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = self.alpha_loss(batch, self.log_ent_coef)

        self.ent_coef_optimizer.zero_grad()
        alpha_loss.backward()
        self.ent_coef_optimizer.step()

        return {
            'train.critic_loss': critic_loss.item(),
            'train.actor_loss': actor_loss.item(),
            'train.alpha_loss': alpha_loss.item(),
            'train.alpha': torch.exp(self.log_ent_coef.detach()).item()
        }

    def select_action(self, state, deterministic=True):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            if len(state.shape) == 1: state = state.unsqueeze(0)
            return self.actor.sample(state, deterministic=deterministic)[0].cpu().numpy()
