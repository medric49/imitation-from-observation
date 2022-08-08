import argparse
import random
import sys

import context_changers
import drqv2
import utils
from pathlib import Path
from metaworld import policies
import dmc

import metaworld_env


env_data = {
    'window_close': ('exp_local/window_close/1/snapshot.pt', 'window-close-v2'),
    'door_open': ('exp_local/door_open/1/snapshot.pt', 'door-open-v2'),
}


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def train(self, *args, **kwargs):
        pass

    def act(self, *args, **kwargs):
        return random.uniform(self.env.action_spec().minimum, self.env.action_spec().maximum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='window_close', type=str, help='Environment name', required=False)
    args, _ = parser.parse_known_args(sys.argv[1:])

    env_dir = args.env
    expert_file, env_name = env_data[env_dir]

    env = metaworld_env.Env(env_name)
    env = dmc.wrap(env, frame_stack=3, action_repeat=2)
    expert = drqv2.DrQV2Agent.load(expert_file)
    expert.train(False)
    agent = RandomAgent(env)

    num_train = 15000
    num_valid = 3000
    ep_len = 45
    video_dir = Path(f'videos/{env_dir}')

    im_w, im_h = 64, 64
    utils.generate_video_from_expert(
        video_dir / 'train/0', expert, env, context_changers.NullContextChanger(), cam_ids=[0],
        ep_len=ep_len, num=num_train, im_w=im_w, im_h=im_h)

    utils.generate_video_from_expert(
        video_dir / 'valid/0', expert, env, context_changers.NullContextChanger(), cam_ids=[0],
        ep_len=ep_len, num=num_valid, im_w=im_w, im_h=im_h)

    utils.generate_video_from_expert(
        video_dir / 'train/1', agent, env, context_changers.NullContextChanger(), cam_ids=[0],
        ep_len=ep_len, num=num_train, im_w=im_w, im_h=im_h)

    utils.generate_video_from_expert(
        video_dir / 'valid/1', agent, env, context_changers.NullContextChanger(), cam_ids=[0],
        ep_len=ep_len, num=num_valid, im_w=im_w, im_h=im_h)



