import gym

def collect_transitions(env, agent, total_timesteps, start_step):
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
        at every timestep
    """
    
    if not isinstance(env, gym.vector.VectorEnv):
        env = gym.vector.SyncVectorEnv([lambda: env])
    
    state = env.reset()
    for step in range(total_timesteps):
        if step < start_step:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)
        
        next_state, reward, done, info = env.step(action)
        
        yield state, action, reward, next_state, done, info
        state = next_state
    
    env.close()