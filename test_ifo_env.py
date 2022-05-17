import os

from gym.envs.mujoco import reacher3dof
from rllab.envs.gym_env import GymEnv

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

env = GymEnv("Reacher3DOF-v1", mode='oracle', force_reset=True)

time_step = env.reset()
print(time_step)
while True:
    env.render()
    time_step = env.step(env.action_space.sample())
    # action = policy(observation)
    # observation, reward, done, info = env.step(action)
    #
    # if done:
    #    observation, info = env.reset(return_info=True)
    print(time_step)
env.close()
