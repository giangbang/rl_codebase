import gym
from gym.vector import SyncVectorEnv
from .utils import get_obs_shape, get_action_dim

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
    env_fns = list(map( lambda e : lambda: e, env ))
    return VecEnv(env_fns)