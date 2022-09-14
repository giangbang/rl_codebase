from typing import Union, Type
from rl_codebase.agents.sac import ContinuousSAC, DiscreteSAC
from rl_codebase.core.buffers import BufferTransition, Transition
from rl_codebase.core.utils import is_image_space, get_obs_shape
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import gym


def _create_dummy_onehot_observation_space(observation_space: gym.spaces, num_envs: int):
    """
    Create an observation space, with onehot representation of tasks
    """
    assert not is_image_space(observation_space), "Not supported observation space"
    observation_dim = get_obs_shape(observation_space)
    observation_shape = (np.prod(observation_dim) + num_envs,)

    low, high = observation_space.low, observation_space.high
    new_low = np.concatenate((low, np.zeros(num_envs, dtype=observation_space.dtype)), axis=-1)
    new_high = np.concatenate((high, np.ones(num_envs, dtype=observation_space.dtype)), axis=-1)

    obs_cls = type(observation_space)
    return obs_cls(new_low, new_high, observation_shape, dtype=observation_space.dtype)


class OnehotSAC(nn.Module):
    """
    SAC agent with multitask, weights are shared between tasks, 
    each task is represented to the agent by an onehot vector
    """

    def __init__(
            self,
            sac_agent_cls: Type[Union[ContinuousSAC, DiscreteSAC]],
            observation_space: gym.spaces,
            action_space: gym.spaces,
            num_envs: int,
            learning_rate: float = 3e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            num_layers=3,
            hidden_dim=256,
            init_temperature=.2,
            device='cpu', ):
        super().__init__()
        self.device = device
        self.num_envs = num_envs
        # Create the dummy onehot shape of the action space
        observation_space = _create_dummy_onehot_observation_space(observation_space, num_envs)
        self.observation_shape = observation_space.shape

        self.sac_agent = sac_agent_cls(
            observation_space=observation_space,
            action_space=action_space,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_temperature=init_temperature
        )

        # Create aliases
        self.actor_optimizer = self.sac_agent.actor_optimizer
        self.critic_optimizer = self.sac_agent.critic_optimizer

        # Each task has a separated entropy coefficient
        self.log_ent_coef = torch.log(init_temperature * torch.ones(self.num_envs,
                                                                    device=device)).requires_grad_(True)

        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef],
                                                   lr=learning_rate,
                                                   )

    def _optimize(self, losses, optimizer):
        """
        Given a list of losses, apply the update to the weights of networks
        """
        losses = sum(losses)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        return losses.item()

    def update(self, batch: BufferTransition):
        critic_losses, actor_losses, alpha_losses = [], [], []

        batch_of_tasks = []
        for i in range(batch.num_tasks):
            batch_of_task = batch.get_task(i)
            batch_of_task = self.concat_onehot_batch(batch_of_task, i, batch.num_tasks)

            batch_of_tasks.append(batch_of_task)

        # Update critic
        for i, batch_of_task in enumerate(batch_of_tasks):
            # critic
            critic_loss = self.sac_agent.critic_loss(batch_of_task, self.log_ent_coef[i])
            critic_losses.append(critic_loss)

        critic_loss = self._optimize(critic_losses, self.critic_optimizer)

        # Update actor
        for i, batch_of_task in enumerate(batch_of_tasks):
            # actor
            actor_loss = self.sac_agent.actor_loss(batch_of_task, self.log_ent_coef[i])
            actor_losses.append(actor_loss)

        actor_loss = self._optimize(actor_losses, self.actor_optimizer)

        # Update alpha
        for i, batch_of_task in enumerate(batch_of_tasks):
            # alpha
            alpha_loss = self.sac_agent.alpha_loss(batch_of_task, self.log_ent_coef[i])
            alpha_losses.append(alpha_loss)

        alpha_loss = sum(alpha_losses) / len(alpha_losses)
        self.ent_coef_optimizer.zero_grad()
        alpha_loss.backward()
        self.ent_coef_optimizer.step()

        alpha_loss = alpha_loss.item()

        return critic_loss, actor_loss, alpha_loss

    def select_action(self, state, deterministic=True):
        state = self.concat_onehot_state(state, list(range(self.num_envs)), self.num_envs)
        return self.sac_agent.select_action(state, deterministic=deterministic)

    def concat_onehot_batch(self, batch: Transition, task: int, num_tasks: int):
        states = self.concat_onehot_state(batch.states, task, num_tasks)
        next_states = self.concat_onehot_state(batch.next_states, task, num_tasks)

        return Transition(states, batch.actions, batch.rewards, next_states, batch.dones)

    def concat_onehot_state(self, state, task, num_tasks: int):
        assert len(state.shape) == 2
        if isinstance(task, int): task = [task]
        if not isinstance(state, torch.Tensor): state = torch.FloatTensor(state)

        onehot = F.one_hot(torch.tensor(task), num_classes=num_tasks).to(self.device)
        broadcast_shape = (state.shape[0], -1)
        onehot = onehot.expand(*broadcast_shape)

        onehot_state = torch.cat((state, onehot), dim=1)
        # Some sanity check
        assert onehot_state[0].shape == self.observation_shape
        return onehot_state
