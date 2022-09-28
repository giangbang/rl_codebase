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
    episode_lengths = np.zeros((env.num_envs,), dtype=float)

    num_episode = 0
    window_size = 50
    rewards_episode_buffer = [deque(maxlen=window_size) for _ in range(env.num_envs)]
    lengths_episode_buffer = [deque(maxlen=window_size) for _ in range(env.num_envs)]
    success_count_buffer = [deque(maxlen=window_size) for _ in range(env.num_envs)]

    has_success_metric = False  # Some envs do not support success measure

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
        reward = np.array(reward)
        done = np.array(done)

        episode_rewards += reward.reshape(episode_rewards.shape)
        episode_lengths += 1

        num_episode += np.sum(done)
        next_state_to_return = copy.deepcopy(next_state)

        for i, d in enumerate(done):
            if d:
                # Record episode rewards
                rewards_episode_buffer[i].extend([episode_rewards[i]])
                episode_rewards[i] = 0

                # Record episode lengths
                lengths_episode_buffer[i].extend([episode_lengths[i]])
                episode_lengths[i] = 0

                # As the VectorEnv resets automatically, `next_state` is already the
                # first observation of the next episode, we need to set it back to
                # the actual final state of the episode
                next_state_to_return[i] = info['final_observation'][i]

                # Measure success rate
                # Several environments do not stop when success,
                # so we only count the success signal when the episode is done
                if 'success' in info or 'is_success' in info:
                    success = info.get('success', info.get('is_success'))
                    success_count_buffer[i].extend([success[i]])

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
            report['train.episodes'] = num_episode
            report['train.rewards'] = np.mean([np.mean(ep_rw) for ep_rw in rewards_episode_buffer])
            report['train.lengths'] = np.mean([np.mean(ep_len) for ep_len in lengths_episode_buffer])
            report['train.success'] = np.mean([np.mean(scc) for scc in success_count_buffer])

        yield (state, action, reward, next_state_to_return, done, info), report
        state = next_state

    env.close()
