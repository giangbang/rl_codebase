import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from rl_codebase.agents.distral.models import DiscreteDistralActor
from rl_codebase.agents.sac import Critic


class DiscreteDistral(nn.Module):
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

        self.actor = DiscreteDistralActor(observation_space, action_space, num_layers,
                                            hidden_dim).to(device)

        self.critic = Critic(observation_space, action_space, num_layers, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic._online_q.parameters(), lr=learning_rate,
        )

    def critic_loss(self, batch, distill_policy: 'DiscreteDistral'):
        # Compute target Q 
        with torch.no_grad():
            next_pi, next_entropy = self.actor.probs(batch.next_states, compute_log_pi=True)

            next_q_vals = self.critic.target_q(batch.next_states)
            next_q_val = torch.minimum(*next_q_vals)

            next_q_val = (next_q_val * next_pi).sum(
                dim=1, keepdims=True
            )

            ent_coef = torch.exp(log_ent_coef)
            next_q_val = next_q_val + ent_coef * next_entropy.reshape(-1, 1)

            target_q_val = batch.rewards + (1 - batch.dones) * self.gamma * next_q_val

        current_q_vals = self.critic.online_q(batch.states)
        current_q_vals = [
            current_q.gather(1, batch.actions)
            for current_q in current_q_vals
        ]
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