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
    'hopper_stand': ('hopper_stand', 'exp_local/hopper_stand/1/snapshot.pt', None, context_changers.NullContextChanger),
    'walker_run': ('walker_run', 'exp_local/walker_run/1/snapshot.pt', None, context_changers.NullContextChanger),
    'finger_turn_easy': ('finger_turn_easy', 'exp_local/finger_turn_easy/1/snapshot.pt', None, context_changers.NullContextChanger),
    'reacher_hard2': ('reacher_hard', 'exp_local/reacher_hard/1/snapshot.pt', 'domain_xmls/reacher.xml', context_changers.ReacherHardWCContextChanger),
    'reacher_hard': ('reacher_hard', 'exp_local/reacher_hard/1/snapshot.pt', 'domain_xmls/reacher.xml', context_changers.ReacherHardContextChanger),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='reacher_hard2', type=str, help='Environment name', required=False)
    parser.add_argument('--episode_len', default=50, type=int, help='Video length', required=False)
    parser.add_argument('--im-w', default=64, type=int, help='Frame width', required=False)
    parser.add_argument('--im-h', default=64, type=int, help='Frame height', required=False)

    args, _ = parser.parse_known_args(sys.argv[1:])

    episode_len = args.episode_len
    task_name = args.env
    env_name, expert_file, xml_file, cc_class = env_data[task_name]

    expert = DrQV2Agent.load(expert_file)
    expert.train(training=False)

    env = dmc.make(env_name, frame_stack=3, action_repeat=2, seed=2, episode_len=episode_len, xml_path=xml_file)
    random_agent = utils.RandomAgent(env)

    num_train = 5000
    num_valid = 400
    im_w, im_h = args.im_w, args.im_h

    video_dir = Path(f'videos/{task_name}')
    utils.generate_video_from_expert(
        video_dir / 'train/0', expert, env, cc_class(), cam_ids=[0], num=num_train, im_w=im_w, im_h=im_h)
    utils.generate_video_from_expert(
        video_dir / 'valid/0', expert, env, cc_class(), cam_ids=[0], num=num_valid, im_w=im_w, im_h=im_h)