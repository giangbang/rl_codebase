from rl_codebase.cmn import (
    get_action_dim,
    get_obs_shape,
    get_observation_space,
    get_action_space
)
import torch
from .models import *

class ContinuousSAC(nn.Module):
    def __init__(
        self, 
        env, 
        learning_rate:float=3e-4, 
        gamma:float=0.99, 
        tau:float=0.005,
        num_layers=3,
        hidden_dim=256,
        init_temperature=.2,
        device='cpu',
    ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.device = device

        observation_space = get_observation_space(env)
        action_space = get_action_space(env)

        action_dim = get_action_dim(action_space)

        self.actor  = ContinuousSACActor(observation_space, action_space, num_layers, 
                hidden_dim).to(device)
         
        self.critic = Critic(obs_shape, action_shape, num_layers, hidden_dim).to(device)
        
        self.target_entropy = -np.prod(action_dim)


        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate, 
        )
        
        self.critic_optimizer = torch.optim.Adam(
            self.critic._online_q.parameters(), lr=learning_rate,
        )
        
        self.log_ent_coef = torch.log(init_temperature*
            torch.ones(1, device=device)).requires_grad_(True)
        
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], 
            lr=learning_rate,
        )

    def _update_critic(self, batch):
        # Compute target Q 
        with torch.no_grad():
            next_pi, next_log_pi  = self.actor.sample(batch.next_states, compute_log_pi=True)
            next_q_vals = self.critic.target_q(batch.next_states, next_pi)
            next_q_val  = torch.minimum(*next_q_vals)
            
            ent_coef    = torch.exp(self.log_ent_coef)
            next_q_val  = next_q_val - ent_coef * next_log_pi
            
            target_q_val= self.reward_scale*batch.rewards + (1-batch.dones)*self.discount*next_q_val
            
        current_q_vals  = self.critic.online_q(batch.states, batch.actions)
        critic_loss     = .5*sum(F.mse_loss(current_q, target_q_val) for current_q in current_q_vals)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.critic.polyak_update(self.critic_tau)
        
        return critic_loss.item()
        
    def _update_actor(self, batch):
        pi, log_pi = self.actor.sample(batch.states, compute_log_pi=True)
        
        q_vals = self.critic.online_q(batch.states, pi)
        q_val  = torch.minimum(*q_vals)
        
        with torch.no_grad():
            ent_coef = torch.exp(self.log_ent_coef)
        
        actor_loss = (ent_coef * log_pi - q_val).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_alpha(self, batch):
        with torch.no_grad():
            pi, log_pi = self.actor.sample(batch.states, compute_log_pi=True)
        alpha_loss = -(self.log_ent_coef * (log_pi + self.target_entropy).detach()).mean()
        
        self.ent_coef_optimizer.zero_grad()
        alpha_loss.backward()
        self.ent_coef_optimizer.step()
        
        return alpha_loss.item()
        
    def update(self, batch):
        
        critic_loss = self._update_critic(batch)
        actor_loss = self._update_actor(batch)
        alpha_loss = self._update_alpha(batch)
        
        return critic_loss, actor_loss, alpha_loss
        
     
    def select_action(self, state, deterministic=True):
        with torch.no_grad():
            return self.actor.sample(state, deterministic=deterministic)[0].cpu().numpy()