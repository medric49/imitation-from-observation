import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch.utils.data
from torch.utils.data.dataset import T_co


class CTVideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, episode_len, cam_ids, same_video=False):
        self._root = Path(root)
        self._files = list(self._root.iterdir())

        self._episode_len = episode_len
        self._same_video = same_video
        self._cam_ids = cam_ids

    def _sample(self):
        if len(self._cam_ids) > 1:
            cam1, cam2 = random.sample(self._cam_ids, k=2)
        else:
            cam1, cam2 = 0, 0

        videos1, videos2 = random.choices(self._files, k=2)

        video1 = np.load(videos1)[cam1, :self._episode_len]
        video2 = np.load(videos1 if self._same_video else videos2)[cam2, :self._episode_len]

        video1 = video1.transpose(0, 3, 1, 2).copy()
        video2 = video2.transpose(0, 3, 1, 2).copy()
        return video1, video2

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self._sample()


class ViRLVideoDataset(torch.utils.data.IterableDataset):

    def __init__(self, root, episode_len, cam_ids):
        self._root = Path(root)
        self._num_classes = len(list(self._root.iterdir()))
        self._files = []

        for c in range(self._num_classes):
            class_dir = self._root / str(c)
            self._files.append(list(class_dir.iterdir()))

        self._episode_len = episode_len
        self._cam_ids = cam_ids

    def _sample(self):
        if len(self._cam_ids) > 1:
            cam1, cam2, cam3 = random.sample(self._cam_ids, k=3)
        else:
            cam1, cam2, cam3 = 0, 0, 0

        classes = list(range(self._num_classes))

        class_1 = random.choice(classes)
        classes.remove(class_1)
        class_2 = random.choice(classes)

        video_i, video_p = random.choices(self._files[class_1], k=2)
        video_n = random.choice(self._files[class_2])

        video_i = np.load(video_i)[cam1, :self._episode_len]
        video_p = np.load(video_p)[cam2, :self._episode_len]
        video_n = np.load(video_n)[cam3, :self._episode_len]

        video_i = video_i.transpose(0, 3, 1, 2).copy()
        video_p = video_p.transpose(0, 3, 1, 2).copy()
        video_n = video_n.transpose(0, 3, 1, 2).copy()

        return video_i, video_p, video_n

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self._sample()
