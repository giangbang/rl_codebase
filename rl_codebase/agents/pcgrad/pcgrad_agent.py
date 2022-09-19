from rl_codebase.core.utils import *
from rl_codebase.agents.sac import ContinuousSAC, DiscreteSAC
from rl_codebase.agents.pcgrad.pcgrad import CorePCGrad
from rl_codebase.agents.sacmt import SACMT
import gym


class PCGrad(SACMT):
    """
    PCGrad, https://arxiv.org/abs/2001.06782
    """
    def __init__(
            self,
            env,
            eval_env=None,
            learning_rate: float = 3e-4,
            buffer_size: int = 1_000_000,  # 1e6
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            num_layers=3,
            hidden_dim=256,
            init_temperature=.2,
            device: str = 'cpu',
            log_path=None,
            **kwargs
    ):
        super().__init__(
            env=env,
            eval_env=eval_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_temperature=init_temperature,
            device=device,
            log_path=log_path,
            **kwargs
        )

        sac_agent_cls = DiscreteSAC if isinstance(self.action_space, gym.spaces.Discrete) else ContinuousSAC
        self.agent = CorePCGrad(
            sac_agent_cls=sac_agent_cls,
            observation_space=self.observation_space,
            action_space=self.action_space,
            num_envs=self.env.num_envs,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_temperature=init_temperature,
            device=device
        )
