import dmc
from bots import ArmEnv
import numpy as np
import cv2

env = ArmEnv()
env = dmc.wrap(env, frame_stack=3, action_repeat=2, episode_len=100)
env.arm.set_ee_pose_components(x=0.30, y=0.1, z=0.20, pitch=np.pi/4, moving_time=2)
