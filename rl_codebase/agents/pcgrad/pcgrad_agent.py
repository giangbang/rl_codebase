from rl_codebase.agents.base import OffPolicyAgent
from rl_codebase.core.utils import *
from rl_codebase.agents.sac import ContinuousSAC, DiscreteSAC
import gym
from rl_codebase.agents.pcgrad.pcgrad import CorePCGrad


class PCGrad(OffPolicyAgent):
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
        super().__init__(env=env, eval_env=eval_env, buffer_size=buffer_size, batch_size=batch_size, device=device,
                         log_path=log_path)

        sac_agent_cls = DiscreteSAC if isinstance(self.action_space, gym.spaces.Discrete) else ContinuousSAC
        self.agent = CorePCGrad(
            sac_agent_cls=sac_agent_cls,
            observation_space=self.observation_space,
            action_space=self.action_space,
            num_envs=env.num_envs,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_temperature=init_temperature,
            device=device
        )

    def update(self, buffer, gradient_steps: int = 1):
        critic_losses, actor_losses, alpha_losses = [], [], []
        alpha = []
        for _ in range(gradient_steps):
            batch = buffer.sample()
            critic_loss, actor_loss, alpha_loss = self.agent.update(batch)

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)
            alpha.append(self.agent.log_ent_coef.exp().mean().detach().cpu().item())

        report = {
            'train.critic_loss': np.mean(critic_losses),
            'train.actor_loss': np.mean(actor_losses),
            'train.alpha_loss': np.mean(alpha_losses),
            'train.alpha': np.mean(alpha)
        }
        return report

    def _select_action(self, state, deterministic=True):
        return self.agent.select_action(state, deterministic=deterministic)

    def set_training_mode(self, mode: bool = False):
        self.agent.train(mode)
