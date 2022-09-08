import torch 
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

from rl_codebase.cmn import (
    create_net,
    get_action_dim,
    get_obs_shape,
    MLP, 
    CNN
)

class ContinuousSACActor(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces,
        action_space: gym.spaces,
        num_layer: int=3,
        hidden_dim=256,
    ):
        super().__init__()
        self.log_std_min    = -10
        self.log_std_max    = 2
        self.action_dim 	= get_action_dim(action_space)

        self.actor = create_net(observation_space, self.action_dim*2,
                num_layer, hidden_dim)

    def forward(self, x):
        return self.actor(x).chunk(2, dim=-1)

    def sample(self, x, compute_log_pi=False, deterministic:bool=False):
        '''
        Sample action from policy, return sampled actions and log prob of that action
        In inference time, set the sampled actions to be deterministic
        
        :param x: observation with type `torch.Tensor`
        :param compute_log_pi: return the log prob of action taken

        '''
        mu, log_std = self.forward(x)
        
        if deterministic: return torch.tanh(mu), None
        
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        Gaussian_distribution = torch.distributions.normal.Normal(
                                mu, log_std.exp())
                                
        sampled_action  = Gaussian_distribution.rsample()
        squashed_action = torch.tanh(sampled_action)
        
        if not compute_log_pi: return squashed_action, None
        
        log_pi_normal   = Gaussian_distribution.log_prob(sampled_action)
        log_pi_normal   = torch.sum(log_pi_normal, dim=-1, keepdim=True)
        
        # See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
        log_squash      = log_pi_normal - torch.sum(
                            torch.log(
                                F.relu(1 - squashed_action ** 2) + 1e-6
                            ),
                            dim = -1, keepdim=True
                        )
        assert len(log_squash.shape) == 2 and len(squashed_action.shape) == 2
        assert log_squash.shape == log_pi_normal.shape
        return squashed_action, log_squash

class DoubleQNet(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces,
        action_space: gym.spaces,
        num_layer: int=3,
        hidden_dim=256,
    ):
        super().__init__()
        state_dim = np.prod(get_obs_shape(observation_space))
        action_dim = get_action_dim(action_space)
        inputs_dim = state_dim + action_dim
        
        self.q1 = MLP(inputs_dim, 1, num_layer, hidden_dim)
        self.q2 = MLP(inputs_dim, 1, num_layer, hidden_dim)

        
    def forward(self, x, a):
        assert x.shape[0] == a.shape[0]
        assert len(x.shape) == 2 and len(a.shape) == 2
        x = torch.cat([x, a], dim=1)
        return self.q1(x), self.q2(x)

class Critic(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces,
        action_space: gym.spaces,
        num_layer: int=3,
        hidden_dim=256,
    ):
        super().__init__()
        
        self._online_q = DoubleQNet(observation_space, action_space, num_layer, hidden_dim)
        self._target_q = DoubleQNet(observation_space, action_space, num_layer, hidden_dim)
        
        for param in self._target_q.parameters():
            param.requires_grad = False
        self._target_q.load_state_dict(self._online_q.state_dict())
        
    def target_q(self, x, a): return self._target_q(x, a)
    
    def online_q(self, x, a): return self._online_q(x, a)
    
    def polyak_update(self, tau):
        '''Exponential evaraging of the online q network'''
        for target, online in zip(self._target_q.parameters(), self._online_q.parameters()):
            target.data.copy_(target.data * (1-tau) + tau * online.data)
    