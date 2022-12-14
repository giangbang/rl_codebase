import os
import gym
import numpy as np
from typing import List
from rl_codebase.core.vec_env import wrap_vec_env
from rl_codebase.core.utils import get_time_now_as_str


def evaluate_policy(env, agent, deterministic: bool = True,
                    num_eval_episodes: int = 10, save_video_to: str = None,
                    report_separated_task: bool = False, task_names: List[str] = None,
                    **kwargs,
                    ):
    """
    Evaluate the `agent` on the given `env`.

    :param env: Environment on which the agent is evaluated, should be an
        instance of `gym.Env` or `gym.VectorEnv`, the latter will be treated as
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
    :param report_separated_task: whether to report seperatedly on each task
        or not, ignore in single-task env.
    ;param task_names: List of the names of tasks, if None, each tasks is named from
        0 to `n_tasks`-1, only works when `report_seperated_task` is set to `True`
    """
    if save_video_to is not None:
        os.makedirs(save_video_to, exist_ok=True)
        save_vid = True
    else:
        save_vid = False

    report = {}

    if not isinstance(env, gym.vector.VectorEnv):
        if save_vid: assert env.render_mode is not None
        env = wrap_vec_env(env)

    num_episodes = np.zeros((env.num_envs,), dtype=int)
    total_return = np.zeros((env.num_envs,), dtype=float)
    success_rate = np.zeros((env.num_envs,), dtype=float)
    ep_len       = np.zeros((env.num_envs,), dtype=float)
    current_ep_len = np.zeros((env.num_envs,), dtype=float)
    current_return = np.zeros((env.num_envs,), dtype=float)

    has_success_metric = False  # Some envs do not support success measure

    state = env.reset()
    if save_vid:
        stop_record = np.zeros((env.num_envs,), dtype=bool)
        frames = [_get_frames_from_VecEnv(env, stop_record)]

    while (num_episodes < num_eval_episodes).any():
        action = agent.select_action(state, deterministic=deterministic)

        next_state, reward, done, info = env.step(action)

        state = next_state
        done = np.array(done)
        reward = np.array(reward)

        current_return += reward
        num_episodes += done
        current_ep_len += 1

        if 'success' in info or 'is_success' in info:
            success = info.get('success', info.get('is_success'))
            has_success_metric = True

        for i, d in enumerate(done):
            if d:
                ep_len[i] += current_ep_len[i]
                current_ep_len[i] = 0

                total_return[i] += current_return[i]
                current_return[i] = 0
            
                if has_success_metric:
                    # Some environments do not halt after `success`
                    # Thus, we only count the `success` at the end of the episode
                    success_rate[i] += success[i]

        if save_vid and not stop_record.all():
            stop_record = np.bitwise_or(done, stop_record)
            frames.append(_get_frames_from_VecEnv(env, stop_record))

    total_return /= num_episodes
    success_rate /= num_episodes
    ep_len /= num_episodes

    if task_names is None:
        task_names = [f'task_{i}' for i in range(env.num_envs)]
    if not isinstance(task_names, list):
        task_names = [task_names]

    # Report separately each task
    if report_separated_task:
        for task_name, reward, length in zip(task_names, total_return, ep_len):
            report[f'eval.{task_name}.rewards'] = reward
            report[f'eval.{task_name}.length'] = length
        if has_success_metric:
            for task_name, success in zip(task_names, success_rate):
                report[f'eval.{task_name}.success'] = success

    # Report average of all tasks
    report['eval.rewards'] = np.mean(total_return)
    report['eval.length'] = np.mean(ep_len)
    if has_success_metric:
        report['eval.success'] = np.mean(success_rate)

    if save_vid:
        tiled_frames = []
        for time_step, frame in enumerate(frames):
            for task, fr in enumerate(frame):
                if fr is None: # End of episode, while other tasks are still running
                    # Patch the missing frames with the last frame of that task
                    frame[task] = frames[time_step-1][task]
            tiled_frames.append(tile_images(frame))

        vid_path = os.path.join(save_video_to, f'{get_time_now_as_str()}.mp4')
        write_video_from_ndarray(tiled_frames, vid_path)

    return report


def _get_frames_from_VecEnv(env, stop_record):
    frames = []
    for e, s in zip(env.envs, stop_record):
        if not s:
            frame = e.render()
            frame = np.array(frame)
            if len(frame.shape) > 3: frame = np.squeeze(frame)
        else:
            frame = None
        frames.append(frame)
    return frames


def write_video_from_ndarray(frames: List[np.ndarray], filename: str):
    import cv2
    assert filename.endswith('.mp4')
    h, w = frames[0].shape[:2]
    fps = 50

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(filename, fourcc, float(fps), (w, h))

    for frame in frames:
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video.write(frame)

    video.release()

def tile_images(img_nhwc: List[np.ndarray]) -> np.ndarray:  # pragma: no cover
    """
    This function is borrowed from Stable-baselines3
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    :param img_nhwc: list or array of images, ndim=4 once turned into array. img nhwc
        n = batch index, h = height, w = width, c = channel
    :return: img_HWc, ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)])
    # img_HWhwc
    out_image = img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image
