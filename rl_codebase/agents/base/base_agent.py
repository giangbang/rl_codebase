from rl_codebase.core import (
    ReplayBuffer,
    evaluate_policy,
    collect_transitions,
    Logger,
    wrap_vec_env,
    get_env_name,
    get_observation_space,
    get_action_space
)

from abc import ABC, abstractmethod
import torch
import gym
import numpy as np


class BaseAgent(ABC):
    def __init__(
            self,
            env,
            eval_env=None,
            log_path=None,
            device='cpu',
            seed=None,
            **kwargs,
    ):
        if not isinstance(env, gym.vector.VectorEnv):
            env = wrap_vec_env(env)

        if eval_env and not isinstance(eval_env, gym.vector.VectorEnv):
            eval_env = wrap_vec_env(eval_env)

        if seed:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.seed(seed=seed)
            
        self.env = env
        self.eval_env = eval_env
        self.num_envs = env.num_envs

        self.observation_space = get_observation_space(env)
        self.action_space = get_action_space(env)

        self.device = device
        self.log_path = log_path
        env_name = get_env_name(env)
        exp_name = self.__class__.__name__
        self.logger = Logger(log_dir=log_path, env_name=env_name, exp_name=exp_name, seed=seed)

    @abstractmethod
    def learn(self,
              total_timesteps: int,
              start_step: int = 1000,
              eval_freq: int = 10000,
              n_eval_episodes: int = 10,
              train_freq: int = 1,
              gradient_steps: int = 1,
              report_separate: bool = False,
              **kwargs,
              ):
        """
        Learn method
        """

    @abstractmethod
    def _select_action(self, state, deterministic: bool = False):
        """
        Select action
        """

    def select_action(self, state, deterministic: bool = False):
        return np.array(self._select_action(state, deterministic=deterministic)).reshape(self.env.action_space.shape)

    @abstractmethod
    def set_training_mode(self, mode: bool = False):
        """
        Set training mode
        """


class OffPolicyAgent(BaseAgent, ABC):
    def __init__(
            self,
            env,
            eval_env=None,
            buffer_size: int = 1_000_000,  # 1e6
            batch_size: int = 256,
            device: str = 'cpu',
            log_path=None, 
            **kwargs,
    ):
        super().__init__(
            env=env,
            eval_env=eval_env,
            log_path=log_path,
            device=device
        )
        self.buffer = ReplayBuffer(
            observation_space = self.observation_space, 
            action_space = self.action_space, 
            capacity = buffer_size,
            batch_size = batch_size, 
            device = device, 
            num_envs = self.env.num_envs
        )

    @abstractmethod
    def update(self, buffer, gradient_steps: int = 1):
        """
        Update weights
        """

    def learn(self,
              total_timesteps: int,
              start_step: int = 1000,
              eval_freq: int = 10000,
              n_eval_episodes: int = 10,
              train_freq: int = 1,
              gradient_steps: int = 1,
              report_separate: bool = False,
              **kwargs,
    ):
        self.set_training_mode(True)
        train_report = {}

        eval_freq = int(eval_freq + self.env.num_envs - 1) // self.env.num_envs

        for step, (transition, time_report) in enumerate(collect_transitions(self.env,
                                                                             self, total_timesteps, start_step)):
            state, action, reward, next_state, done, info = transition

            self.buffer.add(state, action, reward, next_state, done, info)

            if step % train_freq == 0:
                train_report = self.update(self.buffer, gradient_steps)

            if step % eval_freq == 0:
                self.logger.dict_record(time_report)
                self.logger.dict_record(train_report)

                if self.eval_env:
                    self.set_training_mode(False)
                    eval_report = evaluate_policy(self.eval_env, self,
                                                  num_eval_episodes=n_eval_episodes,
                                                  report_separated_task=report_separate)
                    self.set_training_mode(True)
                    self.logger.dict_record(eval_report)
                self.logger.dump()
        self.logger.dump_file()
        self.set_training_mode(False)
