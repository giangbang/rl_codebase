from rl_codebase.cmn import (
    ReplayBuffer,
    evaluate_policy,
    collect_transitions,
    Logger
)
from rl_codebase.cmn.utils import *
from .sac_continuous import ContinuousSAC
import gym
import torch.nn as nn

class SAC:
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

        self.agents = nn.ModuleList([
            ContinuousSAC(env, learning_rate, gamma, tau, num_layers, hidden_dim,
                          init_temperature, device)
            for _ in range(env.num_envs)
        ])
        self.observation_space = get_observation_space(env)
        self.action_space = get_action_space(env)
        self.buffer = ReplayBuffer(self.observation_space, self.action_space, buffer_size,
                                   batch_size, device, env.num_envs)

        self.device = device
        self.log_path = log_path
        self.logger = Logger(log_dir=log_path)

    def select_action(self, state, deterministic: bool = False):
        action = []
        for s, a in zip(state, self.agents):
            ac = a.select_action(state, deterministic=deterministic)
            action.append(ac)
        return np.array(action).reshape(len(action), -1)

    def update(self, buffer):
        critic_losses, actor_losses, alpha_losses = [], [], []
        alpha = []
        batch = buffer.sample()
        for i, a in enumerate(self.agents):
            critic_loss, actor_loss, alpha_loss = a.update(batch.get_task(i))

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)
            alpha.append(a.log_ent_coef.exp().detach().item())

        report = {
            'train.critic_loss': np.mean(critic_losses),
            'train.actor_loss': np.mean(actor_losses),
            'train.alpha_loss': np.mean(alpha_losses),
            'train.alpha': np.mean(alpha)
        }
        return report

    def learn(self,
              total_timesteps,
              start_step: int = 1000,
              eval_freq: int = 10000,
              n_eval_episodes: int = 10,
              train_freq: int = 1,
              ):
        train_report = {}
        for step, (transition, time_report) in enumerate(collect_transitions(self.env,
                                                                             self, total_timesteps, start_step)):
            state, action, reward, next_state, done, info = transition

            self.buffer.add(state, action, reward, next_state, done, info)

            if step % train_freq == 0:
                train_report = self.update(self.buffer)

            if step % eval_freq == 0:
                self.logger.dict_record(time_report)
                self.logger.dict_record(train_report)

                if self.eval_env:
                    eval_report = evaluate_policy(self.eval_env, self,
                                                  num_eval_episodes=n_eval_episodes)
                    self.logger.dict_record(eval_report)
                self.logger.dump()
        self.logger.dump_file()

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

