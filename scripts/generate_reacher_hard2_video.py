import warnings

import context_changers
import dmc
import utils
from drqv2 import DrQV2Agent
from pathlib import Path

warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__ == '__main__':
    expert = DrQV2Agent.load('experts/reacher_hard.pt')
    episode_len = 50
    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=2, episode_len=episode_len, xml_path='domain_xmls/reacher2.xml')
    random_agent = utils.RandomAgent(env)

    num_train = 15000
    num_valid = 3000
    im_w, im_h = 64, 64

    video_dir = Path('videos/reacher_hard2')

    context_changer = context_changers.ReacherHardWCContextChanger()
    utils.generate_video_from_expert(
        video_dir / 'train/1', random_agent, env, context_changer, cam_ids=[0], num=num_train, im_w=im_w, im_h=im_h)

    context_changer = context_changers.ReacherHardWCContextChanger()
    utils.generate_video_from_expert(
        video_dir / 'train/0', expert, env, context_changer, cam_ids=[0], num=num_train, im_w=im_w, im_h=im_h)

    context_changer = context_changers.ReacherHardWCContextChanger()
    utils.generate_video_from_expert(
        video_dir / 'valid/1', random_agent, env, context_changer, cam_ids=[0], num=num_valid, im_w=im_w, im_h=im_h)

    context_changer = context_changers.ReacherHardWCContextChanger()
    utils.generate_video_from_expert(
        video_dir / 'valid/0', expert, env, context_changer, cam_ids=[0], num=num_valid, im_w=im_w, im_h=im_h)



