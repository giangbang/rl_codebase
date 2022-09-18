import torch
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from rl_codebase.agents.sac.models import DiscreteSACActor, Critic


class DiscreteSAC(nn.Module):
    def __init__(
            self,
            observation_space: gym.spaces,
            action_space: gym.spaces,
            learning_rate: float = 3e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            num_layers=3,
            hidden_dim=256,
            init_temperature=1,
            device='cpu',
            target_entropy_ratio=0.2,
    ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = DiscreteSACActor(observation_space, action_space, num_layers,
                                      hidden_dim).to(device)

        self.critic = Critic(observation_space, action_space, num_layers, hidden_dim).to(device)

        self.target_entropy = np.log(action_space.n) * target_entropy_ratio

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
            next_pi, log_probs = self.actor.probs(batch.next_states, compute_log_pi=True)

            next_q_vals = self.critic.target_q(batch.next_states)
            next_q_val = torch.minimum(*next_q_vals)
            
            ent_coef = torch.exp(log_ent_coef)
            
            assert log_probs.shape == next_q_val.shape
            next_q_val = next_q_val - ent_coef * log_probs

            next_q_val = (next_q_val * next_pi).sum(
                dim=1, keepdims=True
            )

            target_q_val = batch.rewards + (1 - batch.dones) * self.gamma * next_q_val

        current_q_vals = self.critic.online_q(batch.states)
        current_q_vals = [
            current_q.gather(1, batch.actions)
            for current_q in current_q_vals
        ]
        critic_loss = .5 * sum(F.mse_loss(current_q, target_q_val) for current_q in current_q_vals)

        return critic_loss

    def actor_loss(self, batch, log_ent_coef):
        pi, log_probs = self.actor.probs(batch.states, compute_log_pi=True)

        with torch.no_grad():
            q_vals = self.critic.online_q(batch.states)
            q_val = torch.minimum(*q_vals)

        with torch.no_grad():
            ent_coef = torch.exp(log_ent_coef)

        q_val = q_val - ent_coef * log_probs
        actor_loss = (pi * q_val).sum(
            dim=1, keepdims=True
        )
        actor_loss = -actor_loss.mean()

        return actor_loss

    def alpha_loss(self, batch, log_ent_coef):
        with torch.no_grad():
            pi, log_probs = self.actor.probs(batch.states, compute_log_pi=True)
            entropy = - (pi * log_probs).sum(dim=1).mean()
        alpha_loss = -(
                log_ent_coef * (-entropy + self.target_entropy).detach()
        ).mean()

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

        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            if len(state.shape) == 1: state = state.unsqueeze(0)
            return self.actor.sample(state, False, deterministic)[0].cpu().numpy()
