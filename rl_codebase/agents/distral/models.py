import torch
import torch.nn as nn
import gym

from rl_codebase.agents.sac import ContinuousSACActor, DiscreteSACActor

def atanh(x: torch.Tensor):
    """
    Inverse of Tanh
    """
    return 0.5 * (x.log1p() - (-x).log1p())

class ContinuousDistralActor(ContinuousSACActor):
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

class DiscreteistralActor(DiscreteSACActor):
    def cross_ent(self, x, action_pi):
        """
        Calculate the cross entropy of the current policy under given `x`
        with the distribution `action_pi`
        """
        logits = self.forward(x)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probs = distribution.logits
        assert action_pi.shape == log_probs.shape
        cross_ent = (action_pi * log_probs).sum(dim=1, keepdim=True)
        return cross_ent