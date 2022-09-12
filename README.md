# rl_codebase
Simple codebase for my RL projects. 
This repo is aimed to be simple, lightweight with minimal amound of code, which serves as a starting point for reproducing RL algorithms. 
Each of the file in the repo is self-contained and loosely coupled with each other for code reusability.
Some part of the code is adapted from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

## Installation
```
pip install git+https://github.com/giangbang/rl_codebase.git
```
or it can be installed manually from code
```
git clone https://github.com/giangbang/rl_codebase.git
cd rl_codebase/
pip install -e .
```

## Implemented algorithm

Some popular RL algorithms have been implemented in `rl_codebase` to provide quick benchmarking.

<<<<<<< HEAD
- [x] SAC
- [ ] Distral
- [ ] PCGrad
=======
- [x] [SAC](https://arxiv.org/pdf/1801.01290): support both continuous and discrete action spaces. When training with multitask environments, each tasks is trained on a separated agent as if they are independent experiments. This serves as a baseline for other multi-task algorithms.
- [ ] Distral
- [x] [PCGrad](https://arxiv.org/pdf/2001.06782)   
>>>>>>> bce7a2c322914a6cfa19ba66a43bc49657d66e8f

### Example

This example show the running of Soft Actor Critic ([SAC](https://arxiv.org/pdf/1812.05905.pdf)) with a few lines of code.
```python
import gym
from rl_codebase import SAC

env  = gym.make('LunarLanderContinuous-v2')
eval_env = gym.make('LunarLanderContinuous-v2', render_mode='rgb_array')

agent = SAC(env, eval_env, log_path='logging')
agent.learn(total_timesteps=1_000_000, start_step=1000, eval_freq=10000)

# Video logging
from rl_codebase import evaluate_policy
evaluate_policy(eval_env, agent, task_names='LunarLanderContinuous',
                save_video_to='video')
```
The codebase follows a sklearn-like syntax and bears much resemblance from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) syntax. But unlike Stable-baselines3, one can have the algorithms run with multi-task environment.
```python
import metaworld
from rl_codebase import SAC
import random
import gym

benchmark = metaworld.MT10()
train_envs, eval_envs = [], []

for name, env_cls in benchmark.train_classes.items():
    train = gym.wrappers.TimeLimit(env_cls(), max_episode_steps=500)
    eval = gym.wrappers.TimeLimit(env_cls(), max_episode_steps=500)
    task = random.choice([task for task in benchmark.train_tasks
                        if task.env_name == name])
    train.set_task(task)
    eval.set_task(task)
    train_envs.append(train)
    eval_envs.append(eval)

agent = SAC(train_envs, eval_envs, log_path='logging')
agent.learn(total_timesteps=1_000_000, start_step=1000, eval_freq=2000)
```
### Logging format
An example of the logger output when training with `PCGrad` on [metaworld](https://github.com/rlworkgroup/metaworld).

```
==============================
time.time_elapsed    : 1296.60
time.total_timesteps : 40000
time.fps             : 30
train.rewards        : 242.09
train.critic_loss    : 0.00
train.actor_loss     : -0.73
train.alpha_loss     : -18.75
train.alpha          : 0.06
eval.rewards         : 494.47
eval.success         : 24.15
```