import os
import time
import cv2
from metaworld.policies import SawyerWindowCloseV2Policy

import metaworld_env

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

env = metaworld_env.Env('window-close-v2')
expert = metaworld_env.Expert(SawyerWindowCloseV2Policy(), env)

time_step = env.reset()
for i in range(env.max_path_length):
    cv2.imshow('frame', cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    a = expert.act(time_step.observation)
    time_step = env.step(a)
    print(time_step.reward)
    time.sleep(1/25.)
