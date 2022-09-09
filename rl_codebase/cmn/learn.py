import gym
from .utils import wrap_vec_env
import sys
import time


def collect_transitions(env, agent, total_timesteps, start_step, eval_freq: int = 1000):
    """
    A simple function to let agent interact with environment.
    Can be helpful for many kind of RL algorithms.
    This function only help interacting and collecting transition, 
    so the bulk of the code is outside this function.
    
    :param env: environment to interact with, should follow a 
        `gym.Env` or `gym.vector.VectorEnv` interface.
    :param agent: agent that interacts with env, should implement a 
        function `select_action` that take input the current `state`
        and a `deterministic` flag
    :param start_step: randomly take action at some first timesteps
    :param total_timesteps: total number of timestep to interact with 
        environment, if `gym.vector.VectorEnv` is provided, each env will be
        run with `total_timesteps` steps, making total of 
        `total_timesteps*num_envs` steps
    :return: a tuple (states, actions, rewards, next_state, done, info)
        at every timestep, and some additional info as python dict (time_elapsed, etc)
    """

    if not isinstance(env, gym.vector.VectorEnv):
        env = wrap_vec_env(env)

    start_time = time.time_ns()
    report = {}

    state = env.reset()
    for step in range(total_timesteps):
        if step < start_step:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        next_state, reward, done, info = env.step(action)

        # report
        if step % eval_freq == 0:
            # We only update training report at every specific intervals
            # to optimize cpu time, 
            time_elapsed = max((time.time_ns() - start_time) / 1e9, sys.float_info.epsilon)
            num_timestep = (step + 1) * env.num_envs
            fps = int(num_timestep / time_elapsed)

            report['time.time_elapsed'] = time_elapsed
            report['time.total_timesteps'] = num_timestep
            report['time.fps'] = fps

        yield (state, action, reward, next_state, done, info), report
        state = next_state

    env.close()
