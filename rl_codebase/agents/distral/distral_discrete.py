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
        
        self.ent_coef = 1 / self.beta
        self.cross_ent_coef = self.alpha / self.beta

    def critic_loss(self, batch, distill_policy: 'DiscreteDistral'):
        # Compute target Q 
        with torch.no_grad():
            next_pi, log_probs = self.actor.probs(batch.next_states, compute_log_pi=True)

            next_q_vals = self.critic.target_q(batch.next_states)
            next_q_val = torch.minimum(*next_q_vals)

            next_q_val = next_q_val - self.ent_coef * log_probs
            next_q_val = (next_q_val * next_pi).sum(
                dim=1, keepdims=True
            )
            
            cross_ent = distill_policy.actor.cross_ent(batch.next_states, next_pi)
            next_q_val = next_q_val + self.cross_ent_coef * cross_ent

            target_q_val = batch.rewards + (1 - batch.dones) * self.gamma * next_q_val

        current_q_vals = self.critic.online_q(batch.states)
        current_q_vals = [
            current_q.gather(1, batch.actions)
            for current_q in current_q_vals
        ]
        critic_loss = .5 * sum(F.mse_loss(current_q, target_q_val) for current_q in current_q_vals)

        return critic_loss

    def log_loss(self, batch, policy_pi: 'DiscreteDistral'):
        with torch.no_grad():
            pi, _ = policy_pi.actor.probs(batch.states, compute_log_pi=False)

        log_distill = self.actor.cross_ent(batch.states, pi)
        log_loss = -self.cross_ent_coef * log_distill.mean()
        return log_loss

    def update_distill(self, batch, policy_pi: 'DiscreteDistral'):
        log_loss = self.log_loss(batch, policy_pi)

        self.actor_optimizer.zero_grad()
        log_loss.backward()
        self.actor_optimizer.step()

        return log_loss.item()
        
    def actor_loss(self, batch, distill_policy: 'DiscreteDistral'):
        pi, log_probs = self.actor.probs(batch.states, compute_log_pi=True)

        with torch.no_grad():
            q_vals = self.critic.online_q(batch.states, pi)
            q_val = torch.minimum(*q_vals)

        cross_ent = distill_policy.actor.cross_ent(batch.states, pi)
        assert log_probs.shape == q_val.shape
        q_val = q_val - self.ent_coef * log_probs
        q_val = (pi * q_val).sum(
            dim=1, keepdims=True
        )

        assert cross_ent.shape == q_val.shape
        actor_loss = (-q_val - self.cross_ent_coef * cross_ent).mean()

        return actor_loss
        
    def update(self, batch, distill_policy: 'DiscreteDistral'):
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
