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
    im_w, im_h = 64, 64

    ep_len = 30

    video_dir = Path('videos/reacher_hard2')

    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=1, xml_path='domain_xmls/reacher_2_targets.xml')
    random_agent = utils.RandomAgent(env)
    context_changer = context_changers.ReacherHardTargetSwitcherContextChanger()
    utils.generate_video_from_expert(
        video_dir / 'train/1', random_agent, env, context_changer, cam_ids=[0],
        ep_len=ep_len, num=num_train // 2, im_w=im_w, im_h=im_h)
    utils.generate_video_from_expert(
        video_dir / 'train/1', expert, env, context_changer, cam_ids=[0],
        ep_len=ep_len, num=num_train // 2, im_w=im_w, im_h=im_h)

    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/reacher2.xml')
    context_changer = context_changers.ReacherHardContextChanger()
    utils.generate_video_from_expert(
        video_dir / 'train/0', expert, env, context_changer, cam_ids=[0],
        ep_len=ep_len, num=num_train, im_w=im_w, im_h=im_h)

    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=3, xml_path='domain_xmls/reacher_2_targets.xml')
    context_changer = context_changers.ReacherHardTargetSwitcherContextChanger()
    utils.generate_video_from_expert(
        video_dir / 'valid/1', expert, env, context_changer, cam_ids=[0],
        ep_len=ep_len, num=num_valid, im_w=im_w, im_h=im_h)

    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=4, xml_path='domain_xmls/reacher2.xml')
    context_changer = context_changers.ReacherHardContextChanger()
    utils.generate_video_from_expert(
        video_dir / 'valid/0', expert, env, context_changer, cam_ids=[0],
        ep_len=ep_len, num=num_valid, im_w=im_w, im_h=im_h)



