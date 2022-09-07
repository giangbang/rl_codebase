import numpy as np
import torch
from collections import namedtuple
import gym
from .utils import *

Transition = namedtuple('Transition',
                        ('states', 'actions', 'rewards', 'next_states', 'dones'))

class BufferTransition(Transition):
    def get_task(self, task: int):
        states = self.states[:, task]
        actions = self.actions[:, task]
        rewards = self.rewards[:, task]
        next_states = self.next_states[:, task]
        dones = self.dones[:, task]
        return Transition(states, actions, rewards, next_states, dones)

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, 
            observation_space: gym.spaces, 
            action_space: gym.spaces, 
            capacity: int=1_000_000, 
            batch_size: int=256, 
            device: str='cpu',
            num_envs: int=None,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        
        if num_envs is None:
            num_envs = 1
            self.is_multitask_buffer = False
        else: 
            self.is_multitask_buffer = True
        
        self.num_envs = num_envs
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.is_image_obs = is_image_space(observation_space)

        obs_shape = self.obs_shape
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = observation_space.dtype

        self.obses = np.empty((capacity, num_envs, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, num_envs, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, num_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.empty((capacity, num_envs, 1), dtype=np.float32)
        self.dones = np.empty((capacity, num_envs, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done, info=None):
        '''Add a new transition to replay buffer'''
        obs = np.array(obs).reshape(self.obses.shape[1:])
        action = np.array(action).reshape(self.actions.shape[1:])
        reward = np.array(reward).reshape(self.rewards.shape[1:])
        next_obs = np.array(next_obs).reshape(self.next_obses.shape[1:])
        done = np.array(done).reshape(self.dones.shape[1:])

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)

        if info is not None:
            timeout_shape = self.dones.shape[1:]
            # [Important] Handle timeout separately for infinite horizon
            timeout = info.get("TimeLimit.truncated", np.zeros(timeout_shape,dtype=bool))
            self.dones[self.idx] *= (1-timeout).reshape(*timeout_shape)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, callback_on_states=None):
        '''
        Sample batch of Transitions with batch_size elements.
        Return a named tuple with 'states', 'actions', 'rewards', 'next_states' and 'dones'. 
        `callback_on_state` is helpful when you want to preprocess the states before feeding to neural nets
        for example, this can be a normalization step, or onehot encoding of tasks
        '''
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        dones = torch.as_tensor(self.dones[idxs], device=self.device)
        
        if callback_on_states is not None:
            obses = callback_on_states(obses)
            next_obses = callback_on_states(next_obses)

        batch_return = BufferTransition(obses, actions, rewards, next_obses, dones)
        
        if not self.is_multitask_buffer: 
            return self._discard_env_dimension(batch_return)
        return batch_return
        
    def _discard_env_dimension(self, transitions: BufferTransition):
        return Transition(*map(lambda x:x.squeeze(axis=1), transitions))