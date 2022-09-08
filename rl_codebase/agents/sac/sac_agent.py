from rl_codebase.cmn import (
    ReplayBuffer,
    evaluate_policy,
    collect_transitions,
    Logger
)
from rl_codebase.cmn.utils import *
from .sac_continuous import ContinuousSAC
import gym

class SAC:
    def __init__(
        self, 
        env,
        eval_env, 
        learning_rate:float=3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
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
            
        self.env = self.env
        self.eval_env = eval_env
        
        self.agents = [
            ContinuousSAC(env, learning_rate, gamma, tau, num_layers, hidden_dim,
                init_temperature)
            for _ in range(env.num_envs)
        ]
        self.observation_space = get_observation_space(env)
        self.action_space = get_action_space(env)
        self.buffer = ReplayBuffer(self.observation_space, self.action_space, buffer_size,
            device, env.num_envs)
            
        self.device = device
        self.log_path = log_path
        self.logger = Logger(log_path)
        
    def select_action(self, state, deterministic:bool=False):
        action = []
        for s, a in zip(state, self.agents):
            ac = a.select_action(state, deterministic=deterministic)
            action.append(ac)
        return np.array(action).reshape(len(action), -1)
        
    def update(self, buffer):
        batch = buffer.sample()
        for i, a in enumerate(self.agents):
            a.update(batch.get_task(i))
    
    def learn(self, 
        total_timesteps,
        start_step: int = 1000,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        train_freq: int=1,
    ):
        for step, (transition, report_train) in enumerate(collect_transitions(self.env,
                self, total_timesteps, start_step)):
            state, action, reward, next_state, done, info = transition
            
            self.buffer.add(state, action, reward, next_state, done, info)
            
            if step % train_freq == 0:
                self.update(self.buffer)
                
            if step % eval_freq == 0:
                self.logger.dict_record(report_train)
                eval_results = evaluate_policy(self.eval_env, self, 
                    num_eval_episodes=n_eval_episodes)
                self.logger.dict_record(eval_results)
                self.logger.dump()
        self.logger.dump_file(self.log_path)