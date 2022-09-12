import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from rl_codebase.agents.distral.models import ContinuousDistralActor
from rl_codebase.agents.sac import Critic


class ContinuousDistral(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces,
                 action_space: gym.spaces,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 num_layers=3,
                 hidden_dim=256,
                 alpha: float = 0.5,# Hyperparam for distral
                 beta: float = 5,   # Hyperparam for distral
                 device='cpu',
                 ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.beta = beta
        self.alpha = alpha

        self.actor = ContinuousDistralActor(observation_space, action_space, num_layers,
                                            hidden_dim).to(device)

        self.critic = Critic(observation_space, action_space, num_layers, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic._online_q.parameters(), lr=learning_rate,
        )

    def critic_loss(self, batch, distill_policy: 'ContinuousDistral'):
        # Compute target Q
        with torch.no_grad():
            next_pi, next_log_pi = self.actor.sample(batch.next_states, compute_log_pi=True)
            next_q_vals = self.critic.target_q(batch.next_states, next_pi)
            next_q_val = torch.minimum(*next_q_vals)

            ent_coef = 1 / self.beta
            next_q_val = next_q_val - ent_coef * next_log_pi

            log_prob_distill = distill_policy.actor.log_probs(batch.next_states, next_pi)
            next_q_val = next_q_val + self.alpha / self.beta * log_prob_distill

            target_q_val = batch.rewards + (1 - batch.dones) * self.gamma * next_q_val

        current_q_vals = self.critic.online_q(batch.states, batch.actions)
        critic_loss = .5 * sum(F.mse_loss(current_q, target_q_val) for current_q in current_q_vals)

        return critic_loss

    def log_loss(self, batch):
        log_distill = self.actor.log_probs(batch.states, batch.actions)
        log_loss = -self.alpha/self.beta * log_distill.mean()
        return log_loss

    def update_distill(self, batch):
        log_loss = self.log_loss(batch)

        self.actor_optimizer.zero_grad()
        log_loss.backward()
        self.actor_optimizer.step()

        return log_loss.item()

    def actor_loss(self, batch, distill_policy: 'ContinuousDistral'):
        pi, log_pi = self.actor.sample(batch.states, compute_log_pi=True)

        q_vals = self.critic.online_q(batch.states, pi)
        q_val = torch.minimum(*q_vals)

        log_prob_distill = distill_policy.actor.log_probs(batch.states, pi)

        actor_loss = (1 / self.beta * log_pi - q_val - self.alpha / self.beta * log_prob_distill).mean()

        return actor_loss

    def update(self, batch, distill_policy: 'ContinuousDistral'):
        # Update critic
        critic_loss = self.critic_loss(batch, distill_policy)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.polyak_update(self.tau)

        # Update actor
        actor_loss = self.actor_loss(batch, distill_policy)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

    def select_action(self, state, deterministic=True):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            if len(state.shape) == 1: state = state.unsqueeze(0)
            return self.actor.sample(state, deterministic=deterministic)[0].cpu().numpy()
