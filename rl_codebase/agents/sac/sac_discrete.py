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
            target_entropy_ratio=0.5,
            adam_eps: float = 1e-8,
            **kwargs,
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
            self.actor.parameters(), lr=learning_rate, eps=adam_eps
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic._online_q.parameters(), lr=learning_rate, eps=adam_eps
        )

        self.log_ent_coef = torch.log(init_temperature *
                                      torch.ones(1, device=device)).requires_grad_(True)

        self.ent_coef_optimizer = torch.optim.Adam(
            [self.log_ent_coef], lr=learning_rate, eps=adam_eps
        )
        
        self.current_policy_entropy = np.log(action_space.n)

    def critic_loss(self, batch, log_ent_coef):
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
        critic_loss = sum(F.mse_loss(current_q, target_q_val) for current_q in current_q_vals)

        return critic_loss

    def actor_loss(self, batch, log_ent_coef):
        pi, ent = self.actor.probs(batch.states, compute_log_pi=True)

        with torch.no_grad():
            q_vals = self.critic.online_q(batch.states)
            q_val = torch.minimum(*q_vals)

        with torch.no_grad():
            ent_coef = torch.exp(log_ent_coef)

        actor_loss = (pi * q_val).sum(
            dim=1, keepdims=True
        ) + ent_coef * ent.reshape(-1, 1)
        actor_loss = -actor_loss.mean()

        return actor_loss

    def alpha_loss(self, batch, log_ent_coef):
        with torch.no_grad():
            pi, entropy = self.actor.probs(batch.states, compute_log_pi=True)
        alpha_loss = -(
                log_ent_coef * (-entropy + self.target_entropy).detach()
        ).mean()
        
        self.current_policy_entropy = torch.mean(entropy).item()

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
            'train.alpha': torch.exp(self.log_ent_coef.detach()).item(),
            'train.entropy': self.current_policy_entropy
        }
        
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            if len(state.shape) == 1: state = state.unsqueeze(0)
            return self.actor.sample(state, False, deterministic)[0].cpu().numpy()
