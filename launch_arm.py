import dmc
from bots import ArmEnv
import numpy as np
import cv2

env = ArmEnv()
env = dmc.wrap(env, frame_stack=3, action_repeat=2, episode_len=100)

time_step = env.reset()

for i in range(1000):
    env.step(np.random.uniform(env.action_spec().minimum, env.action_spec().maximum))
    cv2.imshow('Arm', cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)