from .networks import MLP, CNN
from .utils import *
import torch
import torch.nn as nn
import gym

def create_net(
        observation_space: gym.spaces,
        output_dim: int, 
        num_layer: int=3,
        hidden_dim: int=256,
):
    """
    Quick way to produce a net that a compatible with the observation space
    """
    if is_image_space(observation_space):
        return nn.Sequential(
            CNN(observation_space.shape[-1], hidden_dim),
            MLP(hidden_dim, output_dim, num_layer, hidden_dim//2)
        )
    else:
        state_dim = np.prod(get_obs_shape(observation_space))
        return MLP(state_dim, output_dim, num_layer, hidden_dim)