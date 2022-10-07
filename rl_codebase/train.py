from rl_codebase import *
import argparse

def main():
    parser = argparse.ArgumentParser(description='Training RL agent')

    parser.add_argument('--env_name', default='CartPole-v0')
    parser.add_argument('--algs', default='SAC')
    parser.add_argument('--buffer_size', default=1000000, type=int)

    parser.add_argument('--start_step', default=1000, type=int)
    parser.add_argument('--total_timesteps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--gradient_steps', default=1, type=int)

    parser.add_argument('--n_eval_episodes', default=10, type=int)

    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--num_layers', default=3, type=int)

    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=.5, type=float)

    parser.add_argument('--seed', default=-1, type=int)
    
    args, unknown = parser.parse_known_args()
    unknown = dict(zip(unknown[:-1:2],unknown[1::2]))
    kwargs = vars(args)
    kwargs.update(unknown)
    
    import gym
    from gym import envs

    id = kwargs["env_name"]
    
    try:
        env = gym.make(id, **kwargs)
        eval_env = gym.make(id, **kwargs)
    except:
        from termcolor import colored
        print(colored("Create env from only env id (other params is excluded)", "red"))
        env = gym.make(id)
        eval_env = gym.make(id)

    import sys
    try:
        cls = getattr(sys.modules[__name__], args.algs)
    except:
        raise ValueError(f"{args.algs} is not defined")
    agent = cls(env, eval_env, **kwargs)
    agent.learn(**kwargs)
    agent.save("model", args.total_timesteps)
    
    
if __name__ == "__main__":
    main()