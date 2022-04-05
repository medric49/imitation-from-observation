import dm_env
from dm_env._environment import TimeStep
from dm_env.specs import BoundedArray
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np


class ArmEnv(dm_env.Environment):
    def __init__(self, model_name='rx150'):
        self._env = InterbotixManipulatorXS(model_name, 'arm', 'gripper')
        self._location = None

    def reset(self) -> TimeStep:
        self._location = np.random.uniform([.15, -.25, .18], [.25, .25, .18])
        self._env.arm.set_ee_pose_components(x=self._location[0], y=self._location[1], z=self._location[2], pitch=np.pi/4, moving_time=0.7)
        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST, reward=0., discount=1., observation=None)

    def step(self, action) -> TimeStep:
        self._location += action
        self._location = np.clip(self._location, np.array([.12, -.25, .06]), np.array([.25, .25, .18]))
        self._env.arm.set_ee_pose_components(x=self._location[0], y=self._location[1], z=self._location[2], pitch=np.pi/4, moving_time=0.25)
        return dm_env.TimeStep(step_type=dm_env.StepType.MID, reward=0., discount=1., observation=None)

    def get_random_action(self):
        return np.random.uniform(self.action_spec().minimum, self.action_spec().maximum)

    def observation_spec(self):
        pass

    def action_spec(self):
        return BoundedArray(
            (3,),
            dtype=np.float32,
            minimum=np.array([-.02, -.02, -.02]),
            maximum=np.array([.02, .02, .02])
        )

    def __getattr__(self, name):
        return getattr(self._env, name)
