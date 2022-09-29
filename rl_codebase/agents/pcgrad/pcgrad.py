from typing import Union, Type
from rl_codebase.agents.sac import ContinuousSAC, DiscreteSAC
from rl_codebase.agents.pcgrad.pcgrad_optim import PCGradOptim
from rl_codebase.agents.sacmt import OnehotSAC
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import gym


class CorePCGrad(OnehotSAC):
    """
    Core class for PCGrad, used to manage the training of SAC agent
    """

    def __init__(
            self,
            sac_agent_cls: Type[Union[ContinuousSAC, DiscreteSAC]],
            observation_space: gym.spaces,
            action_space: gym.spaces,
            num_envs: int,
            learning_rate: float = 3e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            device='cpu', 
            **kwargs,
    ):
        super().__init__(
            sac_agent_cls=sac_agent_cls,
            observation_space=observation_space,
            action_space=action_space,
            num_envs=num_envs,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            device=device,
            **kwargs,
        )

        # Wrap with pcgrad optimizer
        self.actor_optimizer = PCGradOptim(self.actor_optimizer)
        self.critic_optimizer = PCGradOptim(self.critic_optimizer)

    def _optimize(self, losses, optimizer):
        optimizer.zero_grad()
        optimizer.pc_backward(losses)
        optimizer.step()

        loss = sum(losses) / len(losses)
        return loss.item()
