import collections
import random

import dm_env
import numpy as np
from dm_env._environment import TimeStep
import metaworld
from dm_env.specs import Array, BoundedArray


class Env(dm_env.Environment):
    def __init__(self, env_name, im_width=84, im_height=84):
        self._observation = None
        env_set = metaworld.MT10()
        self._env = env_set.train_classes[env_name]()
        tasks = [task for task in env_set.train_tasks if task.env_name == env_name]
        task = random.choice(tasks)
        self._env.set_task(task)
        self.im_width = im_width
        self.im_height = im_height

        self.step_id = None
        self._pixels_key = 'pixels'

    def reset(self) -> TimeStep:
        _ = self._env.reset()
        self.step_id = 0

        self._observation = self._env.render(offscreen=True, resolution=(self.im_width, self.im_height))

        observation = collections.OrderedDict()
        observation[self._pixels_key] = np.array(self._observation, dtype=np.uint8)

        return dm_env.TimeStep(dm_env.StepType.FIRST, 0., 1., observation)

    def step(self, action) -> TimeStep:
        obs, reward, done, info = self._env.step(action)
        self.step_id += 1

        self._observation = self._env.render(offscreen=True, resolution=(self.im_width, self.im_height))

        observation = collections.OrderedDict()
        observation[self._pixels_key] = np.array(self._observation, dtype=np.uint8)

        step_type = dm_env.StepType.LAST if self.step_id >= self._env.max_path_length else dm_env.StepType.MID

        return dm_env.TimeStep(step_type, reward, 1., observation)

    def observation_spec(self):
        observation_spec = collections.OrderedDict()
        observation_spec[self._pixels_key] = Array(shape=(self.im_height, self.im_width, 3), dtype=np.uint8,
                                                   name='observation')
        return observation_spec

    def action_spec(self):
        return BoundedArray(shape=self._env.action_space.shape, dtype=self._env.action_space.dtype, minimum=self._env.action_space.low, maximum=self._env.action_space.high)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def render(self):
        return self._env.render(offscreen=True, resolution=(640, 640))


