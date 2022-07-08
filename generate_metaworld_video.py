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


policies = {
    'window-close-v2': policies.SawyerWindowCloseV2Policy
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
    parser.add_argument('--env-name', default='window-close-v2', type=str, help='Environment name', required=False)
    parser.add_argument('--dir-name', default='window_close', type=str, help='Environment name', required=False)
    args, _ = parser.parse_known_args(sys.argv[1:])

    env = metaworld_env.Env(args.env_name)
    env = dmc.wrap(env, frame_stack=3, action_repeat=2)
    # expert = metaworld_env.Expert(policies[args.env_name](), env)
    expert = drqv2.DrQV2Agent.load('exp_local/window_close/1/snapshot.pt')
    expert.train(False)
    agent = RandomAgent(env)

    num_train = 15000
    num_valid = 3000
    ep_len = 50
    video_dir = Path(f'videos/{args.dir_name}')

    im_w, im_h = 64, 64
    utils.generate_video_from_expert(
        video_dir / 'train/0', expert, env, context_changers.NullContextChanger(), cam_ids=[0],
        num_frames=ep_len, num_train=num_train, im_w=im_w, im_h=im_h)

    utils.generate_video_from_expert(
        video_dir / 'valid/0', expert, env, context_changers.NullContextChanger(), cam_ids=[0],
        num_frames=ep_len, num_train=num_valid, im_w=im_w, im_h=im_h)

    utils.generate_video_from_expert(
        video_dir / 'train/1', agent, env, context_changers.NullContextChanger(), cam_ids=[0],
        num_frames=ep_len, num_train=num_train, im_w=im_w, im_h=im_h)

    utils.generate_video_from_expert(
        video_dir / 'valid/1', agent, env, context_changers.NullContextChanger(), cam_ids=[0],
        num_frames=ep_len, num_train=num_valid, im_w=im_w, im_h=im_h)



