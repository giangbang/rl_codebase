from rl_codebase.core import (
    ReplayBuffer,
    evaluate_policy,
    collect_transitions,
    Logger,
    wrap_vec_env,
    get_env_name
)

from abc import ABC, abstractmethod
import gym
import numpy as np


class BaseAgent(ABC):
    def __init__(
        self, 
        env, 
        eval_env=None,
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

        self.device = device
        self.log_path = log_path
        env_name = get_env_name(env)
        exp_name = self.__class__.__name__
        self.logger = Logger(log_dir=log_path, env_name=env_name, exp_name=exp_name)
