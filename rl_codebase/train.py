from rl_codebase import *
import argparse

def main():
    parser = argparse.ArgumentParser(description='Training RL agent')

    parser.add_argument('--env_name', default='CartPole-v0')
    parser.add_argument('--buffer_size', default=1000000, type=int)

    parser.add_argument('--start_step', default=1000, type=int)
    parser.add_argument('--total_env_step', default=1000000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--gradient_steps', default=50, type=int)

    parser.add_argument('--eval_interval', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)

    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--num_layers', default=3, type=int)

    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=.5, type=float)

    parser.add_argument('--seed', default=-1, type=int)
    
    args, unknown = parser.parse_known_args()
    print(unknown)
    
    import gym
    kwargs = vars(args)
    
    env = gym.make()
    
if __name__ == "__main__":
    main()