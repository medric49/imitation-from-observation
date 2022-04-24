
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_modules.core import InterbotixRobotXSCore
import cv2
import threading
import time
import os


import numpy as np

# This script makes the end-effector perform pick, pour, and place tasks
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250'
# Then change to this directory and type 'python bartender.py'




obs_list=[]


def main():



    bot = InterbotixManipulatorXS("rx150", "arm", "gripper")
    camera = cv2.VideoCapture('/dev/video2')
    for i in range(10):
        run_attempt(bot,camera)



def run_attempt(bot,camera):
    bot.gripper.open()
    bot.arm.set_ee_pose_components(x=0.25, y=0, z=0.18, pitch=np.pi / 4, moving_time=1)
    x = threading.Thread(target=observe, args=(bot, camera))
    x.start()
    y = threading.Thread(target=pick, args=(bot, True))
    y.start()
    x.join()
    y.join()

    print(bot.dxl.joint_states.position[6])
    if bot.dxl.joint_states.position[6] > 0.015:
        out = np.array(obs_list)
        print('it grabbed something')
        if os.path.isdir('robotVideos') == False:
            os.system("mkdir robotVideos")
        np.save("robotVideos/video_{a}".format(a=int(time.time() * 1000)), obs_list)
    else:
        print("it grabbed nothing")

    bot.gripper.open()
    bot.arm.set_ee_pose_components(x=0.25, y=0, z=0.18, pitch=np.pi / 4, moving_time=1)
    bot.arm.go_to_sleep_pose()

def get_obs(bot,camera):
    _, observation = camera.read()
    view = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    obs = cv2.resize(view, (64, 64))
    return obs


def observe(bot,camera):
    t1=time.time()
    for i in range(1000):
        obs_list.append(get_obs(bot,camera))
        time.sleep(0.1)
        print('test')
        if time.time()-t1 > 3.55:
            break
    return obs_list




def pick(bot,rand,x=0.25,y=0):
    if rand:
        x=np.random.uniform(0.15,0.25)
        y=np.random.uniform(-0.25,0.25)
    z=0.05
    bot.arm.set_ee_pose_components(x=0.25, y=0, z=0.18, pitch=np.pi / 4, moving_time=1)
    bot.arm.set_ee_pose_components(x=x,y=y,z=z,pitch=np.pi/4,moving_time=1)
    bot.gripper.close(delay=1.5)




if __name__=='__main__':
    main()

'''
    bot = InterbotixManipulatorXS("rx150", "arm", "gripper")
    bot.arm.set_ee_pose_components(x=0.3, z=0.2)
    bot.arm.set_single_joint_position("waist", np.pi/2.0)
    bot.gripper.open()
    bot.arm.set_ee_cartesian_trajectory(x=0.1, z=-0.16)
    bot.gripper.close()
    bot.arm.set_ee_cartesian_trajectory(x=-0.1, z=0.16)
    bot.arm.set_single_joint_position("waist", -np.pi/2.0)
    bot.arm.set_ee_cartesian_trajectory(pitch=1.5)
    bot.arm.set_ee_cartesian_trajectory(pitch=-1.5)
    bot.arm.set_single_joint_position("waist", np.pi/2.0)
    bot.arm.set_ee_cartesian_trajectory(x=0.1, z=-0.16)
    bot.gripper.open()
    bot.arm.set_ee_cartesian_trajectory(x=-0.1, z=0.16)
    bot.arm.go_to_home_pose()
    bot.arm.go_to_sleep_pose()
'''