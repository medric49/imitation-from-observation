import random
import warnings

import context_changers
import dmc
import utils
from drqv2 import DrQV2Agent

warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__ == '__main__':
    env = dmc.make('walker_run', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/walker.xml')
    expert = DrQV2Agent.load('experts/walker_run.pt')

    context_changer = context_changers.WalkerRunContextChanger()

    utils.generate_video_from_expert(
        'videos/walker_run', expert, env, context_changer, cam_ids=[0],
        num_frames=100, num_train=30000, num_valid=6000)

