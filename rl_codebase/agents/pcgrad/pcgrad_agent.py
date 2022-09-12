from rl_codebase.core import (
    ReplayBuffer,
    evaluate_policy,
    collect_transitions,
    Logger,
    wrap_vec_env,
    get_observation_space,
    get_action_space,
)
from rl_codebase.core.utils import *
from rl_codebase.agents.sac import ContinuousSAC, DiscreteSAC
import gym
from .pcgrad import CorePCGrad


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

        self.buffer = ReplayBuffer(self.observation_space, self.action_space, buffer_size,
                                   batch_size, device, env.num_envs)
        self.device = device
        self.log_path = log_path
        self.logger = Logger(log_dir=log_path)

    def update(self, buffer, gradient_steps: int = 1):
        critic_losses, actor_losses, alpha_losses = [], [], []
        alpha = []
        batch = buffer.sample()
        for _ in range(gradient_steps):
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

    def learn(self,
              total_timesteps: int,
              start_step: int = 1000,
              eval_freq: int = 10000,
              n_eval_episodes: int = 10,
              train_freq: int = 1,
              gradient_steps: int = 1,
              ):
        train_report = {}
        eval_freq = int(eval_freq + self.env.num_envs-1) // self.env.num_envs
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
                    eval_report = evaluate_policy(self.eval_env, self,
                                                  num_eval_episodes=n_eval_episodes)
                    self.logger.dict_record(eval_report)
                self.logger.dump()
        self.logger.dump_file()

    def select_action(self, state, deterministic=True):
        return self.agent.select_action(state, deterministic=deterministic).reshape(self.env.action_space.shape)