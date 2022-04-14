import collections

import cv2
import dm_env
from PIL import Image
from dm_env._environment import TimeStep
from dm_env.specs import BoundedArray, Array
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np


class ArmEnv(dm_env.Environment):
    DELTA_MOVE = 0.002
    MOVING_TIME = 0.04

    def __init__(self, model_name='rx150', im_w=84, im_h=84):
        self._env = InterbotixManipulatorXS(model_name, 'arm', 'gripper')
        self._location = None
        self._camera = cv2.VideoCapture('/dev/video2')
        self._im_w = im_w
        self._im_h = im_h
        self._observation = None
        self._view = None

    def update_observation(self):
        _, observation = self._camera.read()
        self._view = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        self._observation = cv2.resize(self._view, (self._im_w, self._im_h))

    def render(self):
        return self._view

    def reset(self) -> TimeStep:
        self._location = np.random.uniform([.15, -.25, .18], [.25, .25, .18])
        self._env.arm.set_ee_pose_components(x=self._location[0], y=self._location[1], z=self._location[2], pitch=np.pi/4, moving_time=1)
        self.update_observation()

        observation = collections.OrderedDict()
        observation['pixels'] = self._observation

        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST, reward=0., discount=1., observation=observation)

    def step(self, action) -> TimeStep:
        action[action < 0] = -ArmEnv.DELTA_MOVE
        action[action > 0] = ArmEnv.DELTA_MOVE
        self._location += action
        self._location = np.clip(self._location, np.array([.15, -.25, .06]), np.array([.35, .25, .18]))
        self._env.arm.set_ee_pose_components(x=self._location[0], y=self._location[1], z=self._location[2], pitch=np.pi/4, moving_time=ArmEnv.MOVING_TIME)

        self.update_observation()

        observation = collections.OrderedDict()
        observation['pixels'] = self._observation

        return dm_env.TimeStep(step_type=dm_env.StepType.MID, reward=0., discount=1., observation=observation)

    def get_random_action(self):
        return np.random.uniform(self.action_spec().minimum, self.action_spec().maximum)

    def observation_spec(self):
        observation_spec = collections.OrderedDict()
        observation_spec['pixels'] = Array(shape=(self._im_h, self._im_w, 3), dtype=np.uint8, name='observation')
        return observation_spec

    def action_spec(self):
        return BoundedArray(
            (3,),
            dtype=np.float32,
            minimum=np.array([-ArmEnv.DELTA_MOVE, -ArmEnv.DELTA_MOVE, -ArmEnv.DELTA_MOVE]),
            maximum=np.array([ArmEnv.DELTA_MOVE, ArmEnv.DELTA_MOVE, ArmEnv.DELTA_MOVE])
        )

    def __getattr__(self, name):
        return getattr(self._env, name)
