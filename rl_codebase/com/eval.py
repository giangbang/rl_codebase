import os
import gym
import numpy as np

def evaluate_policy(env, agent, deterministic: bool=True, 
	num_eval_episodes: int=10, save_video_to:str=None,
	report_seperated_task: bool=False, task_names: list[str]=None
):
	"""
	Evaluate the `agent` on the given `env`.

	:param env: Environment on which the agent is evaluated, should be an
	 	instance of `gym.Env` or `gym.VecEnv`, the latter will be treated as 
	 	multitask environment.
	:param agent: agent, should implement a function `select_action`, which 
		takes two inputs: the current `state` and a flag `deterministic`.
	:param deterministic: use a deterministic policy to evaluate
	:param save_video_to: dir to the folder for video saving, no video recording 
		if `None`.
	:param num_eval_episodes: number of evaluation environment episode, 
		in case the environment is multi-task, each task will be evaluated
		with `num_eval_episodes` episodes, making a total of 
		`num_eval_episodes * n_tasks` env episodes.
	:param report_seperated_task: whether to report seperatedly on each task
		or not, ignore in single-task env.
	;param task_names: List of the names of tasks, if None, each tasks is named from
		0 to `n_tasks`-1, only works when `report_seperated_task` is set to `True`
	"""
	if save_video_to is not None:
		os.makedirs(save_video_to, exists_ok=True)
		save_vid = True
	else:
		save_vid = False

	report = {}

	if isinstance(env, gym.Env):
		env = gym.vector.SyncVectorEnv([lambda: env])

	num_episodes = np.zeros((env.num_envs,), dtype=int)
	total_return = np.zeros((env.num_envs,), dtype=float)

	state = env.reset()
	if save_vid:
		stop_record = np.zeros((env.num_envs,), dtype=bool)
		frames = [ _get_frames_from_VecEnv(env, stop_record) ]

	while (num_episodes < num_eval_episodes).any():
		action = agent.select_action(state, deterministic=deterministic)

		next_state, reward, done, info = env.step(action)
        
        total_return += reward * (num_episodes < num_eval_episodes)
        num_episodes += done

        if save_vid:
        	frames.append( _get_frames_from_VecEnv(env, stop_record) )
        	stop_record = np.bitwise_or(done, stop_record)

    total_return /= num_eval_episodes

    if task_names is None:
    	task_names = [f'task_{i}' for i in range(env.num_envs)] 

    if report_seperated_task:
    	for task_name, reward in zip(task_names, total_return):
    		report[f'eval.{task_name}.rewards'] = reward
   	else:
   		report['eval.rewards'] = np.mean(total_return)

   	if save_vid:
   		frames = list(zip(*frames))
   		for task_name, fr in zip(task_names, frames):
   			vid_path = os.path.join(save_video_to, f'{task_name}.mp4')
   			write_video_from_ndarray(fr, vid_path)

   	return report


def _get_frames_from_VecEnv(env, stop_record):
	frames = []
	for e, s in zip(env.envs, stop_record):
		frame = e.render('rgb_array') if not s else None
		frames.append(frame)
	return frames

def write_video_from_ndarray(frames: list[np.ndarray], filename:str):
	w, h = 150, 125
	fps = 50

	fourcc = cv2.VideoWriter_fourcc(*'MP42') 
	video = cv2.VideoWriter(filename, fourcc, float(fps), (w, h))

	for frame in frames:
		if frame is not None:
			video.write(frame)

	video.release()