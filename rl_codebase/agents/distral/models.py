import torch
import torch.nn as nn
import gym

from rl_codebase.agents.sac import ContinuousSACActor

def atanh(x: torch.Tensor):
    """
    Inverse of Tanh
    """
    return 0.5 * (x.log1p() - (-x).log1p())

class ContinuousDistralActor(ContinuousSACActor):
    def __init__(
            self,
            observation_space: gym.spaces,
            action_space: gym.spaces,
            num_layer: int = 3,
            hidden_dim=256,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            num_layer=num_layer,
            hidden_dim=hidden_dim
        )

    def log_probs(self, x, squashed_action):
        """
        Compute log prob of the action a
        """
        mu, log_std = self.forward(x)
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        distribution = torch.distributions.normal.Normal(
            mu, log_std.exp())

        action = atanh(squashed_action)
        log_pi_normal = distribution.log_prob(action)
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
        return log_squash

