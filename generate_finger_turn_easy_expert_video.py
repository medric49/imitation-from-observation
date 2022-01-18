import warnings

import utils

warnings.filterwarnings('ignore', category=DeprecationWarning)


if __name__ == '__main__':
    utils.generate_video_from_expert('videos/finger_turn_easy', 'experts/finger_turn_easy.pt', 'finger_turn_easy',
                                     num_train=800,
                                     num_valid=200,
                                     xml_path='domain_xmls/finger.xml')
