import warnings

import utils

warnings.filterwarnings('ignore', category=DeprecationWarning)


if __name__ == '__main__':
    utils.generate_video_from_expert('videos/finger_turn_easy', 'experts/finger_turn_easy.pt', 'finger_turn_easy',
                                     cam_ids=[2, 3, 4, 5, 6],
                                     num_frames=15,
                                     num_train=800,
                                     num_valid=200,
                                     xml_path='domain_xmls/finger.xml')
