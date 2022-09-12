from rl_codebase.core import (
    ReplayBuffer,
    evaluate_policy,
    collect_transitions,
    Logger,
    wrap_vec_env,
    get_observation_space,
    get_action_space,
)
import torch
import gym
import numpy as np
import torch.nn as nn
from rl_codebase.agents.distral.distral_continuous import ContinuousDistral


class Distral:
    def __init__(
            self,
            env,
            eval_env=None,
            learning_rate: float = 3e-4,
            buffer_size: int = 1_000_000,  # 1e6
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: float = 0.5,
            beta: float = 5,
            num_layers=3,
            hidden_dim=256,
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

        distral_kwargs = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            alpha=alpha,
            beta=beta,
            device=device
        )

        self.agents = nn.ModuleList([
            ContinuousDistral(**distral_kwargs)
            for _ in range(env.num_envs)
        ])

        self.distill_agent = ContinuousDistral(**distral_kwargs)

        self.buffer = ReplayBuffer(self.observation_space, self.action_space, buffer_size,
                                   batch_size, device, env.num_envs)
        self.device = device
        self.log_path = log_path
        self.logger = Logger(log_dir=log_path)

    def update(self, buffer, gradient_steps: int = 1):
        critic_losses, actor_losses= [], []

        for _ in range(gradient_steps):
            batch = buffer.sample()
            for i, a in enumerate(self.agents):
                task_batch = batch.get_task(i)
                critic_loss, actor_loss = a.update(task_batch, self.distill_agent)
                self.distill_agent.update_distill(task_batch)

                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
        report = {
            'train.critic_loss': np.mean(critic_losses),
            'train.actor_loss': np.mean(actor_losses),
        }
        return report

    def learn(self,
              total_timesteps: int,
              start_step: int = 1000,
              eval_freq: int = 10000,
              n_eval_episodes: int = 10,
              train_freq: int = 1,
              gradient_steps: int = 1,
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
                                                  num_eval_episodes=n_eval_episodes)
                    self.set_training_mode(True)
                    self.logger.dict_record(eval_report)
                self.logger.dump()
        self.logger.dump_file()
        self.set_training_mode(False)

    def set_training_mode(self, mode: bool) -> None:
        self.agents.train(mode)
        self.distill_agent.train(mode)

    def select_action(self, state, deterministic: bool = False):
        action = []
        for s, a in zip(state, self.agents):
            ac = a.select_action(s, deterministic=deterministic)
            action.append(ac)
        return np.array(action).reshape(self.env.action_space.shape)
