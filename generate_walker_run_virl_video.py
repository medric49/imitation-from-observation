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
    env = dmc.make('walker_run', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/walker.xml')
    expert = DrQV2Agent.load('experts/walker_run.pt')
    agent = RandomAgent(env)

    num_train = 15000
    num_valid = 3000
    ep_len = 100
    video_dir = Path('videos/walker_run_virl')

    utils.generate_video_from_expert(
        video_dir / 'train/1', agent, env, context_changers.WalkerRunContextChanger(), cam_ids=[0],
        num_frames=ep_len, num_train=num_train)

    utils.generate_video_from_expert(
        video_dir / 'valid/1', agent, env, context_changers.WalkerRunContextChanger(), cam_ids=[0],
        num_frames=ep_len, num_train=num_valid)

    utils.generate_video_from_expert(
        video_dir / 'train/0', expert, env, context_changers.WalkerRunContextChanger(), cam_ids=[0],
        num_frames=ep_len, num_train=num_train)

    utils.generate_video_from_expert(
        video_dir / 'valid/0', expert, env, context_changers.WalkerRunContextChanger(), cam_ids=[0],
        num_frames=ep_len, num_train=num_valid)

