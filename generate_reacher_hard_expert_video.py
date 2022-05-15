import random
import warnings

import context_changers
import dmc
import utils
from drqv2 import DrQV2Agent

warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__ == '__main__':
    env = dmc.make('reacher_hard', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/reacher.xml')
    expert = DrQV2Agent.load('experts/reacher_hard.pt')

    context_changer = context_changers.ReacherHardContextChanger()

    utils.generate_video_from_expert(
        'videos/reacher_hard', expert, env, context_changer, cam_ids=[0],
        num_frames=30, num_train=3000, num_valid=600)

