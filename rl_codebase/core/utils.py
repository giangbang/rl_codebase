# Some of the functions in this file are borrowed from stable-baselines3
from typing import Dict, Tuple, Union, List
from gym import spaces
from datetime import datetime
import gym
import numpy as np


def get_obs_shape(
        observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    From Stable-baselines3
    Get the shape of the observation (useful for the buffers).
    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_action_dim(action_space: spaces.Space) -> int:
    """
    From Stable-baselines3
    Get the dimension of the action space.
    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def is_image_space(observation_space: spaces.Space) -> bool:
    """
    Check if a observation space has the shape, limits and dtype
    of a valid image.
    The check is conservative, so that it returns False if there is a doubt.
    """
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if observation_space.dtype != np.uint8:
            return False

        # Check the value range
        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            return False
    return False


def get_observation_space(env: gym.Env):
    if isinstance(env, gym.vector.VectorEnv):
        return env.single_observation_space
    return env.observation_space


def get_action_space(env: gym.Env):
    if isinstance(env, gym.vector.VectorEnv):
        return env.single_action_space
    return env.action_space


def get_time_now_as_str():
    return datetime.now().strftime('%Y-%m-%d.%H%M%S')


def get_env_name(env: gym.Env):
    if isinstance(env, gym.vector.VectorEnv):
        # Access the first sub-environment
        env = env.envs[0]
    return str(env.unwrapped.__class__.__name__)
