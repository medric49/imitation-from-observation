import dm_env
from dm_env._environment import TimeStep
from dm_env.specs import BoundedArray
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np


class ArmEnv(dm_env.Environment):
    def __init__(self, model_name='rx150'):
        self.bot = InterbotixManipulatorXS(model_name, 'arm', 'gripper')

    def reset(self) -> TimeStep:
        np.random.uniform()
        self.bot.arm.set_joint_positions(np.random.uniform([-np.pi/3, -np.pi/20, -np.pi/12, np.pi/3, 0.], [np.pi/3, 0., 0., np.pi/2, 0.]))
        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST, reward=0., discount=1., observation=None)

    def step(self, action) -> TimeStep:
        ac = np.zeros((5,))
        ac[:4] = action
        self.bot.arm.set_joint_positions(ac)
        return dm_env.TimeStep(step_type=dm_env.StepType.MID, reward=0., discount=1., observation=None)

    def get_random_action(self):
        return np.random.uniform([-np.pi/3, -np.pi/20, -np.pi/12, np.pi/3], [np.pi/3, np.pi/20, np.pi/20, np.pi/2])

    def observation_spec(self):
        pass

    def action_spec(self):
        return BoundedArray(
            (4,),
            dtype=np.float32,
            minimum=np.array([-np.pi/3, -np.pi/20, -np.pi/12, np.pi/3]),
            maximum=np.array([np.pi/3, np.pi/20, np.pi/20, np.pi/2])
        )
