import random
import warnings

import context_changers
import dmc
import utils
from drqv2 import DrQV2Agent
from pathlib import Path

warnings.filterwarnings('ignore', category=DeprecationWarning)


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def train(self, training):
        pass

    def act(self, obs, step, eval_mode):
        return random.uniform(self.env.action_spec().minimum, self.env.action_spec().maximum)


if __name__ == '__main__':
    env = dmc.make('hopper_hop', frame_stack=3, action_repeat=2, seed=2)
    expert = DrQV2Agent.load('exp_local/hopper_hop/1/snapshot.pt')
    agent = RandomAgent(env)

    num_train = 15000
    num_valid = 3000
    ep_len = 50
    video_dir = Path('videos/hopper_hop')
    im_w, im_h = 64, 64

    utils.generate_video_from_expert(
        video_dir / 'train/1', agent, env, context_changers.NullContextChanger(), cam_ids=[0],
        ep_len=ep_len, num=num_train, im_w=im_w, im_h=im_h)

    utils.generate_video_from_expert(
        video_dir / 'valid/1', agent, env, context_changers.NullContextChanger(), cam_ids=[0],
        ep_len=ep_len, num=num_valid, im_w=im_w, im_h=im_h)

    utils.generate_video_from_expert(
        video_dir / 'train/0', expert, env, context_changers.NullContextChanger(), cam_ids=[0],
        ep_len=ep_len, num=num_train, im_w=im_w, im_h=im_h)

    utils.generate_video_from_expert(
        video_dir / 'valid/0', expert, env, context_changers.NullContextChanger(), cam_ids=[0],
        ep_len=ep_len, num=num_valid, im_w=im_w, im_h=im_h)

