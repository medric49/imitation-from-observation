import warnings

import dmc
import utils
from drqv2 import DrQV2Agent

warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__ == '__main__':
    env = dmc.make('finger_turn_easy', frame_stack=3, action_repeat=2, seed=2, xml_path='domain_xmls/finger.xml')
    expert = DrQV2Agent.load('experts/finger_turn_easy.pt')
    utils.generate_video_from_expert(
        'videos/finger_turn_easy', expert, env,
        cam_ids=[2, 3, 4, 5, 6], ep_len=15, num=800, num_valid=200)
