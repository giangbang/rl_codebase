import torch
import gym
import numpy as np
import torch.nn as nn
from rl_codebase.agents.base import OffPolicyAgent
from rl_codebase.agents.distral.distral_continuous import ContinuousDistral
from rl_codebase.agents.distral.distral_discrete import DiscreteDistral


class Distral(OffPolicyAgent):
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
        super().__init__(env=env, eval_env=eval_env, buffer_size=buffer_size, batch_size=batch_size, device=device,
                         log_path=log_path)

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

        distral_cls = DiscreteDistral if isinstance(self.action_space, gym.spaces.Discrete) else ContinuousDistral

        self.agents = nn.ModuleList([
            distral_cls(**distral_kwargs)
            for _ in range(env.num_envs)
        ])

        self.distill_agent = distral_cls(**distral_kwargs)

    def update(self, buffer, gradient_steps: int = 1):
        critic_losses, actor_losses = [], []

        for _ in range(gradient_steps):
            batch = buffer.sample()
            for i, a in enumerate(self.agents):
                task_batch = batch.get_task(i)
                critic_loss, actor_loss = a.update(task_batch, self.distill_agent)
                self.distill_agent.update_distill(task_batch, a)

                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
        report = {
            'train.critic_loss': np.mean(critic_losses),
            'train.actor_loss': np.mean(actor_losses),
        }
        return report

    def set_training_mode(self, mode: bool = False) -> None:
        self.agents.train(mode)
        self.distill_agent.train(mode)

    def _select_action(self, state, deterministic: bool = False):
        action = []
        for s, a in zip(state, self.agents):
            ac = a.select_action(s, deterministic=deterministic)
            action.append(ac)
        return action
