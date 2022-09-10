from rl_codebase.cmn import (
    ReplayBuffer,
    evaluate_policy,
    collect_transitions,
    Logger,
    wrap_vec_env
)
from rl_codebase.cmn.utils import *
from .sac_continuous import ContinuousSAC
from .sac_discrete import DiscreteSAC
import gym
import torch.nn as nn


class PCGrad:
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
    ):
        if not isinstance(env, gym.vector.VectorEnv):
            env = wrap_vec_env(env)

        if eval_env and not isinstance(eval_env, gym.vector.VectorEnv):
            eval_env = wrap_vec_env(eval_env)

        self.env = env
        self.eval_env = eval_env

        self.observation_space = get_observation_space(env)
        self.action_space = get_action_space(env)