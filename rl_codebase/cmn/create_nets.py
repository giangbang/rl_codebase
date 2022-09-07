from .networks import MLP, CNN
from .utils import *
import torch
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