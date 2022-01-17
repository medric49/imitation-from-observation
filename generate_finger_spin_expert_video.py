import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import dmc
import utils
from drqv2 import DrQV2Agent

warnings.filterwarnings('ignore', category=DeprecationWarning)


if __name__ == '__main__':
    video_dir = Path('videos/finger_spin')
    video_dir.mkdir(parents=True, exist_ok=False)

    agent = DrQV2Agent.load('experts/finger_spin.pt')
    agent.train(training=False)

    im_w, im_h = 64, 64
    env = dmc.make('finger_spin', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/finger.xml')

    def spinner(time_step):
        action = agent.act(time_step.observation, 1, eval_mode=True)
        return action

    def make_video(index, parent_dir):
        parent_dir.mkdir()

        cam2 = []
        cam3 = []
        cam4 = []
        cam5 = []
        cam6 = []
        time_step = env.reset()
        cam2.append(env.physics.render(im_w, im_h, camera_id=2))
        cam3.append(env.physics.render(im_w, im_h, camera_id=3))
        cam4.append(env.physics.render(im_w, im_h, camera_id=4))
        cam5.append(env.physics.render(im_w, im_h, camera_id=5))
        cam6.append(env.physics.render(im_w, im_h, camera_id=6))
        while not time_step.last():
            action = spinner(time_step)
            time_step = env.step(action)
            cam2.append(env.physics.render(im_w, im_h, camera_id=2))
            cam3.append(env.physics.render(im_w, im_h, camera_id=3))
            cam4.append(env.physics.render(im_w, im_h, camera_id=4))
            cam5.append(env.physics.render(im_w, im_h, camera_id=5))
            cam6.append(env.physics.render(im_w, im_h, camera_id=6))
        videos = np.array([cam2, cam3, cam4, cam5, cam6], dtype=np.uint8)
        np.save(parent_dir / f'{index}', videos)

    with torch.no_grad():
        for i in tqdm(range(800)):
            make_video(i, video_dir / 'train')
        for i in tqdm(range(200)):
            make_video(i, video_dir / 'valid')