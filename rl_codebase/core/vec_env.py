import gym
import numpy as np
from gym.vector import SyncVectorEnv
from rl_codebase.core.utils import get_obs_shape, get_action_dim


class DummyVecEnv(SyncVectorEnv):
    def reset(self):
        obs_shape = self.single_observation_space.shape
        states = []
        for env in self.envs:
            states.append(env.reset())
        return np.array(states).reshape(self.num_envs, *obs_shape)

    def step(self, actions):
        obs_shape = self.single_observation_space.shape
        infos_key = set()
        next_states, rewards, dones, infos = [], [], [], []
        for i, (env , action) in enumerate(self.envs, actions):
            next_state, reward, done, info = env.step(action)
            
            if done:
                info.update({'final_observation': next_state})
                next_state = env.reset()
            
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            infos_key.update(info.keys())
        
        info_return = {}
        for key in infos_key:
            val = []
            for inf in infos:
                val.append(inf.get(key, None))
            info_return[key] = np.array(val)
            
        return np.array(next_states).reshape(self.num_envs, *obs_shape),
               np.array(rewards).reshape(self.num_envs, -1),
               np.array(dones).reshape(self.num_envs, -1),
               info_return
        

class VecEnv(SyncVectorEnv):
    """
    `VectorEnv` class of openai gym does not allow the difference
    in `observation_space` of environments being batched, this is 
    not necessarily desirable in multi-task environments, since 
    different env can have different range of the observation states.
    Metaworld env is one example. We only check shape of observation
    and action in this class.
    """

    def _check_spaces(self) -> bool:
        single_obs_shape = get_obs_shape(self.single_observation_space)
        single_action_shape = get_action_dim(self.single_action_space)

        for env in self.envs:
            if not (get_obs_shape(env.observation_space) ==
                    single_obs_shape):
                raise RuntimeError(
                    "Some environments have observation shape different from"
                    f"`{single_obs_shape}`."
                )

            if not (get_action_dim(env.action_space) ==
                    single_action_shape):
                raise RuntimeError(
                    "Some environments have action shape different from"
                    f"`{single_action_shape}`."
                )

        return True


def wrap_vec_env(env):
    if isinstance(env, gym.vector.VectorEnv):
        return env
    if not isinstance(env, list):
        env = [env]
    env_fns = list(map(lambda e: lambda: e, env))
    return VecEnv(env_fns)
