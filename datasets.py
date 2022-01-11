from typing import Iterator

import torch
import torch.utils.data
from torch.utils.data.dataset import T_co
from pathlib import Path
import numpy as np
import random

import utils


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, root):
        self._root = Path(root)
        self._files = list(self._root.iterdir())

    def __iter__(self) -> Iterator[T_co]:
        cam1, cam2 = random.sample([0, 1, 2, 3], k=2)

        videos_1, videos_2 = random.choices(self._files, k=2)

        video_1 = np.load(videos_1)[cam1]
        video_2 = np.load(videos_2)[cam2]
        yield video_1, video_2


if __name__ == '__main__':
    dataset = VideoDataset('videos/finger/train')
    print('ici')
    print(next(iter(dataset))[0].shape )
