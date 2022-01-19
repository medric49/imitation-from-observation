import warnings

import utils

warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__ == '__main__':
    utils.generate_video_from_expert('videos/reacher_hard', 'experts/reacher_hard.pt', 'reacher_hard',
                                     cam_ids=[0],
                                     num_train=2000,
                                     num_valid=300)
