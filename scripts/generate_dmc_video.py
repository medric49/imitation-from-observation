import argparse
import warnings
import sys

import context_changers
import dmc
import utils
from drqv2 import DrQV2Agent
from pathlib import Path

warnings.filterwarnings('ignore', category=DeprecationWarning)

env_data = {
    'reacher_hard2': ('reacher_hard', 'experts/reacher_hard.pt', 'domain_xmls/reacher.xml', context_changers.ReacherHardWCContextChanger),
    'reacher_hard': ('reacher_hard', 'experts/reacher_hard.pt', 'domain_xmls/reacher.xml', context_changers.ReacherHardContextChanger),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='reacher_hard', type=str, help='Environment name', required=False)
    parser.add_argument('--ep_len', default=50, type=int, help='Video length', required=False)
    args, _ = parser.parse_known_args(sys.argv[1:])

    episode_len = args.ep_len
    task_name = args.env
    env_name, expert_file, xml_file, cc_class = env_data[task_name]

    expert = DrQV2Agent.load(expert_file)
    expert.train(training=False)

    env = dmc.make(env_name, frame_stack=3, action_repeat=2, seed=2, episode_len=episode_len, xml_path=xml_file)
    random_agent = utils.RandomAgent(env)

    num_train = 5000
    num_valid = 400
    im_w, im_h = 64, 64

    video_dir = Path(f'videos/{task_name}')

    utils.generate_video_from_expert(
        video_dir / 'train/1', random_agent, env, cc_class(), cam_ids=[0], num=num_train, im_w=im_w, im_h=im_h)
    utils.generate_video_from_expert(
        video_dir / 'train/0', expert, env, cc_class(), cam_ids=[0], num=num_train, im_w=im_w, im_h=im_h)
    utils.generate_video_from_expert(
        video_dir / 'valid/1', random_agent, env, cc_class(), cam_ids=[0], num=num_valid, im_w=im_w, im_h=im_h)
    utils.generate_video_from_expert(
        video_dir / 'valid/0', expert, env, cc_class(), cam_ids=[0], num=num_valid, im_w=im_w, im_h=im_h)
