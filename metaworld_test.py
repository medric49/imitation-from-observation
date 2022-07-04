import os
import metaworld
import random
import time
import cv2
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from matplotlib import pyplot as plt

env_set = metaworld.MT10()

print(env_set.train_classes)
env_name = 'peg-insert-side-v2'

env = env_set.train_classes[env_name]()
tasks = [task for task in env_set.train_tasks if task.env_name == env_name]
# print(len(tasks))
# task = random.choice(tasks)
task = tasks[5]
env.set_task(task)

print(env.action_space)


obs = env.reset()
for i in range(env.max_path_length):
    cv2.imshow('frame', env.render(offscreen=True, resolution=(640, 640)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    a = env.action_space.sample()
    obs, reward, done, info = env.step(a)
    print(reward)
    time.sleep(1/25.)

