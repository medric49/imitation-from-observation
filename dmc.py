import random
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
import os

import torch

import context_changers
import ct_model
import drqv2
import utils

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
from dm_env._environment import TimeStep
from hydra.utils import to_absolute_path


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class EncodeStackWrapper(dm_env.Environment):
    def __init__(self, env, expert, context_translator, expert_env, context_camera_ids, n_video, im_w, im_h, state_dim, frame_stack, dist_reward):
        self._env = env

        self.context_changer = context_changers.ReacherHardContextChanger()
        self.expert: drqv2.DrQV2Agent = expert
        self.expert.train(False)
        self.context_translator: ct_model.CTNet = context_translator
        self.context_translator.eval()
        self.frame_stack = frame_stack

        self.expert_env = expert_env

        self.context_camera_ids = context_camera_ids
        self.n_video = n_video

        self.im_w = im_w
        self.im_h = im_h
        self.state_dim = state_dim
        self.dist_reward = dist_reward

        self.avg_states = None
        self.avg_frames = None

        self.init_channel = self._env.observation_spec().shape[0] // self.frame_stack

        self.step_id = None

    def make_expert_video(self):
        with torch.no_grad():
            videos = []
            for _ in range(self.n_video):
                self.context_changer.reset()

                cam_id = random.choice(self.context_camera_ids)
                episode = []
                time_step = self.expert_env.reset()

                with utils.change_context(self.expert_env, self.context_changer):
                    episode.append(self.expert_env.physics.render(self.im_w, self.im_h, camera_id=cam_id))
                while not time_step.last():
                    action = self.expert.act(time_step.observation, 1, eval_mode=True)
                    time_step = self.expert_env.step(action)
                    with utils.change_context(self.expert_env, self.context_changer):
                        episode.append(self.expert_env.physics.render(self.im_w, self.im_h, camera_id=cam_id))
                videos.append(episode)
            videos = np.array(videos, dtype=np.uint8)
            videos = videos.transpose((0, 1, 4, 2, 3))
        return videos

    def predict_avg_states_frames(self, fobs):
        expert_videos = self.make_expert_video()
        with torch.no_grad():
            states = []
            frames = []

            fobs = torch.tensor(fobs, device=utils.device(), dtype=torch.float)
            expert_videos = torch.tensor(expert_videos, device=utils.device(), dtype=torch.float)
            for expert_video in expert_videos:
                state, frame = self.context_translator.translate(expert_video, fobs, keep_enc2=False)
                states.append(state)
                frames.append(frame)
            states = torch.stack(states)  # n x T x z
            frames = torch.stack(frames)  # n x T x c x h x w

            avg_states = states.mean(dim=0)  # T x z
            avg_frames = frames.mean(dim=0)  # T x c x h x w

        avg_states = avg_states.cpu().numpy()
        avg_frames = avg_frames.cpu().numpy()

        return avg_states, avg_frames

    def encode(self, observation):
        frames = []
        for i in range(self.frame_stack):
            index = i * self.init_channel
            frames.append(observation[index: index + self.init_channel, :, :])
        frames = np.array(frames)
        frames = torch.tensor(frames, device=utils.device(), dtype=torch.float)
        with torch.no_grad():
            state = self.context_translator.encode(frames)
            state = state.view((-1,)).cpu().numpy()
        return state

    def reset(self) -> TimeStep:
        time_step = self._env.reset()

        fobs = time_step.observation[-self.init_channel:]
        self.avg_states, self.avg_frames = self.predict_avg_states_frames(fobs)
        self.step_id = 0

        state = self.encode(time_step.observation)
        target_state = self.avg_states[self.step_id + 1]
        state = np.concatenate([state, target_state])

        return time_step._replace(observation=state, reward=0.)

    def step(self, action) -> TimeStep:
        time_step = self._env.step(action)

        self.step_id += 1

        state = self.encode(time_step.observation)
        s0 = state[-self.state_dim:]

        if time_step.last():
            target_state = self.avg_states[self.step_id]
        else:
            target_state = self.avg_states[self.step_id + 1]
        observation = np.concatenate([state, target_state])

        if self.dist_reward:
            reward = -np.linalg.norm(s0 - target_state)
        else:
            reward = time_step.reward

        return time_step._replace(observation=observation, reward=reward)

    def observation_spec(self):
        return specs.Array(shape=(self.state_dim * (self.frame_stack + 1),), dtype=np.float32, name='observation')

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ChangeContextWrapper(dm_env.Environment):
    def __init__(self, env, context_changer, camera_id, im_h, im_w, pixels_key):
        self._context_changer = context_changer
        self._env = env
        self._camera_id = camera_id
        self._im_h = im_h
        self._im_w = im_w
        self._pixels_key = pixels_key

    def reset(self):
        self._context_changer.reset()
        time_step = self._env.reset()
        self._context_changer.change_env(self._env)
        observation = time_step.observation
        observation[self._pixels_key] = self._env.physics.render(height=self._im_h, width=self._im_w,
                                                                 camera_id=self._camera_id)
        time_step = time_step._replace(observation=observation)
        return time_step

    def step(self, action):
        time_step = self._env.step(action)
        self._context_changer.change_env(self._env)
        observation = time_step.observation
        observation[self._pixels_key] = self._env.physics.render(height=self._im_h, width=self._im_w,
                                                                 camera_id=self._camera_id)
        time_step = time_step._replace(observation=observation)
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class EpisodeLenWrapper(dm_env.Environment):
    def __init__(self, env, ep_len):
        self._env = env
        self._ep_len = ep_len
        self._counter = 0

    def reset(self) -> TimeStep:
        self._counter = 0
        return self._env.reset()

    def step(self, action) -> TimeStep:
        self._counter += 1
        time_step = self._env.step(action)
        if self._counter == self._ep_len:
            time_step = time_step._replace(step_type=StepType.LAST)
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed, xml_path=None, camera_id=None, im_w=84, im_h=84, context_changer=None, episode_len=None):
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'

    if xml_path is not None:
        env.physics.reload_from_xml_path(to_absolute_path(xml_path))

    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        if camera_id is None:
            camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=im_h, width=im_w, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
        if context_changer is not None:
            env = ChangeContextWrapper(env, context_changer, camera_id, im_h, im_w, pixels_key)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    if episode_len is not None:
        env = EpisodeLenWrapper(env, episode_len)
    return env


def wrap(env, frame_stack, action_repeat, episode_len=None):
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = FrameStackWrapper(env, frame_stack, 'pixels')
    env = ExtendedTimeStepWrapper(env)
    if episode_len is not None:
        env = EpisodeLenWrapper(env, episode_len)
    return env

