import random
import warnings

import context_changers
import dmc
import utils
from drqv2 import DrQV2Agent

warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__ == '__main__':
    expert = DrQV2Agent.load('experts/reacher_hard.pt')

    num_train = 15000
    num_valid = 3000

    ep_len = 30

    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/reacher_2_targets.xml')
    context_changer = context_changers.ReacherHardTargetSwitcherContextChanger()
    utils.generate_video_from_expert(
        'videos/reacher_hard_virl/train/1', expert, env, context_changer, cam_ids=[0],
        num_frames=ep_len, num_train=num_train)

    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/reacher.xml')
    context_changer = context_changers.ReacherHardContextChanger()
    utils.generate_video_from_expert(
        'videos/reacher_hard_virl/train/0', expert, env, context_changer, cam_ids=[0],
        num_frames=ep_len, num_train=num_train)

    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/reacher_2_targets.xml')
    context_changer = context_changers.ReacherHardTargetSwitcherContextChanger()
    utils.generate_video_from_expert(
        'videos/reacher_hard_virl/valid/1', expert, env, context_changer, cam_ids=[0],
        num_frames=ep_len, num_train=num_valid)

    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/reacher.xml')
    context_changer = context_changers.ReacherHardContextChanger()
    utils.generate_video_from_expert(
        'videos/reacher_hard_virl/valid/0', expert, env, context_changer, cam_ids=[0],
        num_frames=ep_len, num_train=num_valid)



