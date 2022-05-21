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

    def __init__(self, root, episode_len, cam_ids, augmentation=True):
        self._root = Path(root)
        self._num_classes = len(list(self._root.iterdir()))
        self._files = []
        self._augmentation = augmentation

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

        if self._augmentation:
            video_i, video_p, video_n = self.add_noise(video_i, video_p, video_n)
            video_i, video_p, video_n = self.random_shuffle(video_i, video_p, video_n)
            video_i, video_p, video_n = self.random_sequence_cropping(video_i, video_p, video_n)

        video_i = video_i.transpose(0, 3, 1, 2).copy()
        video_p = video_p.transpose(0, 3, 1, 2).copy()
        video_n = video_n.transpose(0, 3, 1, 2).copy()

        return video_i, video_p, video_n

    def add_noise(self, video_i, video_p, video_n, mean=128., std=0.02):
        video_1 = video_i + np.random.normal(mean, std, video_i.shape)
        video_2 = video_p + np.random.normal(mean, std, video_p.shape)
        video_3 = video_n + np.random.normal(mean, std, video_n.shape)

        video_1 = np.clip(video_1, 0., 255.)
        video_2 = np.clip(video_2, 0., 255.)
        video_3 = np.clip(video_3, 0., 255.)

        return video_1, video_2, video_3

    def random_shuffle(self, video_i, video_p, video_n, p=0.5):
        shuffle = np.random.rand() > p
        if shuffle:
            video_n = video_i.copy()
            np.random.shuffle(video_n)
        return video_i, video_p, video_n

    def random_sequence_cropping(self, video_i, video_p, video_n):
        T = video_i.shape[0]
        base = sum(list(range(T)))
        p_list = [(T - i)/base for i in range(T)]

        video_1, video_2, video_3 = [], [], []
        for i in range(T):
            keep = np.random.rand() > p_list[i]

            if keep:
                video_1.append(video_i[i])
                video_2.append(video_p[i])
                video_3.append(video_n[i])

        video_1 = np.stack(video_1)
        video_2 = np.stack(video_2)
        video_3 = np.stack(video_3)
        return video_1, video_2, video_3

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self._sample()
