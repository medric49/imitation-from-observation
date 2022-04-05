import dm_env
from dm_env._environment import TimeStep
from dm_env.specs import BoundedArray
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np


class ArmEnv(dm_env.Environment):
    def __init__(self, model_name='rx150'):
        self._env = InterbotixManipulatorXS(model_name, 'arm', 'gripper')

    def reset(self) -> TimeStep:
        self._env.arm.set_ee_pose_components(
            x=np.random.uniform(0.15, 0.25),
            y=np.random.uniform(-0.25, 0.25),
            z=.18,
            pitch=np.pi/4
        )
        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST, reward=0., discount=1., observation=None)

    def step(self, action) -> TimeStep:
        self._env.arm.set_ee_pose_components(x=action[0], y=action[1], z=action[2], pitch=np.pi / 4)
        return dm_env.TimeStep(step_type=dm_env.StepType.MID, reward=0., discount=1., observation=None)

    def get_random_action(self):
        return np.random.uniform(self.action_spec().minimum, self.action_spec().maximum)

    def observation_spec(self):
        pass

    def action_spec(self):
        return BoundedArray(
            (3,),
            dtype=np.float32,
            minimum=np.array([.12, -.25, .06]),
            maximum=np.array([.25, .25, .18])
        )

    def __getattr__(self, name):
        return getattr(self._env, name)
