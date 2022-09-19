import torch
import torch.nn as nn
import gym
import numpy as np

from rl_codebase.core import (
    create_net,
    get_action_dim,
    get_obs_shape,
    MLP,
)


class DiscreteSACActor(nn.Module):
    def __init__(
            self,
            observation_space: gym.spaces,
            action_space: gym.spaces,
            num_layer: int = 3,
            hidden_dim=256,
    ):
        super().__init__()
        assert isinstance(action_space, gym.spaces.Discrete)
        action_dim = action_space.n
        self.actor = create_net(observation_space, action_dim,
                                num_layer, hidden_dim)

    def forward(self, x):
        return self.actor(x)

    def probs(self, x, compute_log_pi=False):
        logits = self.forward(x)
        distribution = torch.distributions.Categorical(logits=logits)
        if not compute_log_pi: return distribution.probs, None
        return distribution.probs, distribution.entropy()

    def sample(self, x, compute_log_pi=False, deterministic: bool = False):
        logits = self.forward(x)
        distribution = torch.distributions.Categorical(logits=logits)
        if deterministic: return torch.argmax(distribution.probs, dim=1), None

        sampled_action = distribution.sample()

        if not compute_log_pi: return sampled_action, None

        entropy = distribution.entropy()
        return sampled_action, entropy


class ContinuousSACActor(nn.Module):
    def __init__(
            self,
            observation_space: gym.spaces,
            action_space: gym.spaces,
            num_layer: int = 3,
            hidden_dim=256,
    ):
        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.action_dim = get_action_dim(action_space)

        self.actor = create_net(observation_space, self.action_dim * 2,
                                num_layer, hidden_dim)

    def forward(self, x):
        return self.actor(x).chunk(2, dim=-1)

    def sample(self, x, compute_log_pi=False, deterministic: bool = False):
        """
        Sample action from policy, return sampled actions and log prob of that action
        In inference time, set the sampled actions to be deterministic

        :param x: observation with type `torch.Tensor`
        :param compute_log_pi: return the log prob of action taken

        """
        mu, log_std = self.forward(x)

        if deterministic: return torch.tanh(mu), None

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        Gaussian_distribution = torch.distributions.normal.Normal(
            mu, log_std.exp())

        sampled_action = Gaussian_distribution.rsample()
        squashed_action = torch.tanh(sampled_action)

        if not compute_log_pi: return squashed_action, None

        log_pi_normal = Gaussian_distribution.log_prob(sampled_action)
        log_pi_normal = torch.sum(log_pi_normal, dim=-1, keepdim=True)

        # See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
        log_squash = log_pi_normal - torch.sum(
            torch.log(
                1 - squashed_action ** 2 + 1e-6
            ),
            dim=-1, keepdim=True
        )
        assert len(log_squash.shape) == 2 and len(squashed_action.shape) == 2
        assert log_squash.shape == log_pi_normal.shape
        return squashed_action, log_squash


class DualQNet(nn.Module):
    """
    Dual Q network for discrete action space
    arXiv:1509.06461
    """
    def __init__(
        self,
        inputs_dim: int, 
        outputs_dim: int, 
        n_layer: int, 
        n_unit: int = 256, 
        activation_fn=nn.Tanh,
    ):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim

        net = [nn.Linear(inputs_dim, n_unit), activation_fn()]
        for _ in range(n_layer - 2):
            net.append(nn.Linear(n_unit, n_unit))
            net.append(activation_fn())
        self.v_head = nn.Linear(n_unit, 1)
        self.a_head = nn.Linear(n_unit, outputs_dim)

        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        a = self.a_head(x)
        v = self.v_head(x)
        return v + a - a.mean(dim=1, keepdim=True)


def create_q_net(inputs_dim, output_dim, num_layer, hidden_dim, use_dual_qnet=True):
    if use_dual_qnet:
        return DualQNet(inputs_dim, output_dim, num_layer, hidden_dim)
    else:
        return MLP(inputs_dim, output_dim, num_layer, hidden_dim)

class DoubleQNet(nn.Module):
    def __init__(
            self,
            observation_space: gym.spaces,
            action_space: gym.spaces,
            num_layer: int = 3,
            hidden_dim=256,
    ):
        super().__init__()
        state_dim = np.prod(get_obs_shape(observation_space))
        action_dim = get_action_dim(action_space)

        self.is_discrete_action = isinstance(action_space, gym.spaces.Discrete)
        # If the action space is continuous, action are concatenated with state in Q function
        inputs_dim = state_dim + action_dim * (1 - self.is_discrete_action)
        # If the action space is discrete, output all the action Q values of a state
        output_dim = 1 if not self.is_discrete_action else action_space.n

        if self.is_discrete_action:
            self.q1 = create_q_net(inputs_dim, output_dim, num_layer, hidden_dim, True)
            self.q2 = create_q_net(inputs_dim, output_dim, num_layer, hidden_dim, True)
        else:
            self.q1 = MLP(inputs_dim, output_dim, num_layer, hidden_dim)
            self.q2 = MLP(inputs_dim, output_dim, num_layer, hidden_dim)

    def forward(self, x, a=None):
        if not self.is_discrete_action:
            assert x.shape[0] == a.shape[0]
            assert len(x.shape) == 2 and len(a.shape) == 2
            x = torch.cat([x, a], dim=1)
        return self.q1(x), self.q2(x)


class Critic(nn.Module):
    def __init__(
            self,
            observation_space: gym.spaces,
            action_space: gym.spaces,
            num_layer: int = 3,
            hidden_dim=256,
    ):
        super().__init__()

        self._online_q = DoubleQNet(observation_space, action_space, num_layer, hidden_dim)
        self._target_q = DoubleQNet(observation_space, action_space, num_layer, hidden_dim)

        for param in self._target_q.parameters():
            param.requires_grad = False
        self._target_q.load_state_dict(self._online_q.state_dict())

    def target_q(self, x, a=None):
        return self._target_q(x, a)

    def online_q(self, x, a=None):
        return self._online_q(x, a)

    def polyak_update(self, tau):
        """Exponential averaging of the online q network"""
        with torch.no_grad():
            for target, online in zip(self._target_q.parameters(), self._online_q.parameters()):
                target.data.mul_(1 - tau)
                torch.add(target.data, online.data, alpha=tau, out=target.data)