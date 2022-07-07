import random
import sys
import warnings

import argparse

import context_changers
import dmc
import metaworld_env
import utils
from drqv2 import DrQV2Agent
from pathlib import Path
from metaworld import policies

warnings.filterwarnings('ignore', category=DeprecationWarning)


policies = {
    'window-close-v2': policies.SawyerWindowCloseV2Policy
}


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def train(self, training):
        pass

    def act(self, obs, step, eval_mode):
        return random.uniform(self.env.action_spec().minimum, self.env.action_spec().maximum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='window-close-v2', type=str, help='Environment name', required=False)
    args, _ = parser.parse_known_args(sys.argv[1:])

    env = metaworld_env.Env(args.env_name)
    env = dmc.wrap(env, frame_stack=1, action_repeat=1)
    expert = metaworld_env.Expert(policies[args.env_name](), env)
    agent = RandomAgent(env)

    num_train = 15000
    num_valid = 3000
    ep_len = 100
    video_dir = Path(f'videos/{args.env_name}')

    im_w, im_h = 224, 224
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



