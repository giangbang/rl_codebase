from rl_codebase.core.utils import *
from rl_codebase.agents.base import OffPolicyAgent
from .sac_continuous import ContinuousSAC
from .sac_discrete import DiscreteSAC
import gym
import torch.nn as nn
import torch


class SAC(OffPolicyAgent):
    """
    Vanilla Soft Actor-Critic algorithm
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
            device: str = 'cpu',
            log_path=None,
            **kwargs,
    ):
        super().__init__(env=env, eval_env=eval_env, buffer_size=buffer_size, batch_size=batch_size, device=device,
                         log_path=log_path, **kwargs,)

        agent_cls = DiscreteSAC if isinstance(self.action_space, gym.spaces.Discrete) else ContinuousSAC
        self.agents = nn.ModuleList([
            agent_cls(observation_space=self.observation_space,
                      action_space=self.action_space,
                      learning_rate=learning_rate,
                      gamma=gamma,
                      tau=tau,
                      num_layers=num_layers,
                      hidden_dim=hidden_dim,
                      device=device,
                      **kwargs, # This might includes `init_temperature`
            )
            for _ in range(self.env.num_envs)
        ])

    def _select_action(self, state, deterministic: bool = False):
        action = []
        for s, a in zip(state, self.agents):
            ac = a.select_action(s, deterministic=deterministic)
            action.append(ac)
        return action

    def update(self, buffer, gradient_steps: int = 1):
        critic_losses, actor_losses, alpha_losses = [], [], []
        alpha = []
        for _ in range(gradient_steps):
            batch = buffer.sample()
            for i, a in enumerate(self.agents):
                task_batch = batch.get_task(i)
                critic_loss, actor_loss, alpha_loss = a.update(task_batch)

                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
                alpha_losses.append(alpha_loss)
                alpha.append(a.log_ent_coef.exp().detach().cpu().item())

        report = {
            'train.critic_loss': np.mean(critic_losses),
            'train.actor_loss': np.mean(actor_losses),
            'train.alpha_loss': np.mean(alpha_losses),
            'train.alpha': np.mean(alpha)
        }
        return report

    def save(self, model_dir, step):
        import os
        os.makedirs(model_dir, exist_ok=True)

        torch.save(
            self.agents.state_dict(), '%s/SACAgent_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.agents.load_state_dict(
            torch.load('%s/SACAgent_%s.pt' % (model_dir, step))
        )

    def set_training_mode(self, mode: bool = False) -> None:
        self.agents.train(mode)
