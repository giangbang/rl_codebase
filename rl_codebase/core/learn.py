import gym
from rl_codebase.core.vec_env import wrap_vec_env
import sys
import time
import numpy as np
from collections import deque
import copy


def collect_transitions(env, agent, total_timesteps, start_step, eval_freq: int = 1000):
    """
    A simple function to let agent interact with environment.
    Can be helpful for many kind of RL algorithms.
    This function only help to interact and collect transition,
    so the bulk of the code is outside this function.
    
    :param env: environment to interact with, should follow a 
        `gym.Env` or `gym.vector.VectorEnv` interface.
    :param agent: agent that interacts with env, should implement a 
        function `select_action` that take input the current `state`
        and a `deterministic` flag
    :param start_step: randomly take action at some first timesteps
    :param total_timesteps: total number of timestep to interact with 
        environment, if `gym.vector.VectorEnv` is provided, each env will be
        run with `total_timesteps/num_envs` steps (round up to nearest integer), 
        making total of ~ `total_timesteps` steps.
    :return: a tuple (states, actions, rewards, next_state, done, info)
        at every timestep, and some additional info as python dict (time_elapsed, etc)
    """

    if not isinstance(env, gym.vector.VectorEnv):
        env = wrap_vec_env(env)

    start_time = time.time_ns()
    episode_rewards = np.zeros((env.num_envs,), dtype=float)
    rewards_episode_buffer = [deque(maxlen=50) for _ in range(env.num_envs)]

    report = {}
    state = env.reset()

    total_timesteps = int(total_timesteps + env.num_envs - 1) // env.num_envs
    eval_freq = int(eval_freq + env.num_envs - 1) // env.num_envs

    for step in range(total_timesteps + 1):
        if step < start_step:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        next_state, reward, done, info = env.step(action)
        episode_rewards += reward
        next_state_to_return = next_state

        for i, d in enumerate(done):
            if d:
                rewards_episode_buffer[i].extend([episode_rewards[i]])
                episode_rewards[i] = 0

        # As the VectorEnv resets automatically, `next_state` is already the
        # first observation of the next episode
        if 'final_observation' in info:
            next_state_to_return = copy.deepcopy(next_state)
            final_obs_indx = info['_final_observation']
            final_obs = np.array(list(info['final_observation'][final_obs_indx]), dtype=state.dtype)
            next_state_to_return[final_obs_indx] = final_obs

        # report
        if step % eval_freq == 0:
            # We only update training report at every specific intervals
            # to optimize cpu time, 
            time_elapsed = max((time.time_ns() - start_time) / 1e9, sys.float_info.epsilon)
            num_timestep = step * env.num_envs
            fps = int(num_timestep / time_elapsed)

            report['time.time_elapsed'] = time_elapsed
            report['time.total_timesteps'] = num_timestep
            report['time.fps'] = fps
            report['train.rewards'] = np.mean([np.mean(ep_rw) for ep_rw in rewards_episode_buffer])

        yield (state, action, reward, next_state_to_return, done, info), report
        state = next_state

    env.close()