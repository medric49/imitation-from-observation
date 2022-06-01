import random
import warnings

import context_changers
import dmc
import utils
from drqv2 import DrQV2Agent
from pathlib import Path

warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__ == '__main__':
    expert = DrQV2Agent.load('experts/reacher_hard.pt')

    num_train = 15000
    num_valid = 3000

    ep_len = 30

    video_dir = Path('videos/reacher_hard_virl_v2')

    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/reacher_2_targets.xml')
    context_changer = context_changers.ReacherHardTargetSwitcherContextChanger()
    utils.generate_video_from_expert(
        video_dir / 'train/1', expert, env, context_changer, cam_ids=[0],
        num_frames=ep_len, num_train=num_train)

    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/reacher.xml')
    context_changer = context_changers.ReacherHardContextChanger()
    utils.generate_video_from_expert(
        video_dir / 'train/0', expert, env, context_changer, cam_ids=[0],
        num_frames=ep_len, num_train=num_train)

    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/reacher_2_targets.xml')
    context_changer = context_changers.ReacherHardTargetSwitcherContextChanger()
    utils.generate_video_from_expert(
        video_dir / 'valid/1', expert, env, context_changer, cam_ids=[0],
        num_frames=ep_len, num_train=num_valid)

    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/reacher.xml')
    context_changer = context_changers.ReacherHardContextChanger()
    utils.generate_video_from_expert(
        video_dir / 'valid/0', expert, env, context_changer, cam_ids=[0],
        num_frames=ep_len, num_train=num_valid)



