import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch.utils.data
from torch import nn
from torch.utils.data.dataset import T_co
from torch.nn import functional as F


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


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

    @staticmethod
    def augment(video_i: torch.Tensor, video_p: torch.Tensor, video_n: torch.Tensor):
        # video_i, video_p, video_n = ViRLVideoDataset.augment_images(video_i, video_p, video_n)
        # video_i, video_p, video_n = ViRLVideoDataset.add_noise(video_i, video_p, video_n)
        video_i, video_p, video_n = ViRLVideoDataset.random_shuffle(video_i, video_p, video_n)
        video_i, video_p, video_n = ViRLVideoDataset.random_sequence_cropping(video_i, video_p, video_n)
        return video_i, video_p, video_n

    @staticmethod
    def augment_images(video_i: torch.Tensor, video_p: torch.Tensor, video_n: torch.Tensor):
        aug = RandomShiftsAug(pad=8)
        T = video_i.shape[0]
        video_1, video_2, video_3 = [], [], []
        for t in range(T):
            frame_1, frame_2, frame_3 = video_i[t], video_p[t], video_n[t]
            frame_1, frame_2, frame_3 = aug(frame_1), aug(frame_2), aug(frame_3)
            video_1.append(frame_1)
            video_2.append(frame_2)
            video_3.append(frame_3)
        video_1 = torch.stack(video_1)
        video_2 = torch.stack(video_2)
        video_3 = torch.stack(video_3)
        return video_1, video_2, video_3

    @staticmethod
    def add_noise(video_i: torch.Tensor, video_p: torch.Tensor, video_n: torch.Tensor, mean=128., std=0.02):
        video_1 = video_i + torch.normal(mean, std, video_i.shape, device=video_i.device)
        video_2 = video_p + torch.normal(mean, std, video_p.shape, device=video_p.device)
        video_3 = video_n + torch.normal(mean, std, video_n.shape, device=video_n.device)

        video_1 = torch.clip(video_1, 0., 255.)
        video_2 = torch.clip(video_2, 0., 255.)
        video_3 = torch.clip(video_3, 0., 255.)

        return video_1, video_2, video_3

    @staticmethod
    def random_shuffle(video_i: torch.Tensor, video_p: torch.Tensor, video_n: torch.Tensor, p=0.5):
        shuffle = np.random.rand() < p
        if shuffle:
            indx = torch.randperm(video_i.shape[0])
            video_i = video_i[indx]
            video_p = video_p[indx]
            video_n = video_n[indx]
        return video_i, video_p, video_n

    @staticmethod
    def random_sequence_cropping(video_i: torch.Tensor, video_p: torch.Tensor, video_n: torch.Tensor):
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

        video_1 = torch.stack(video_1)
        video_2 = torch.stack(video_2)
        video_3 = torch.stack(video_3)
        return video_1, video_2, video_3

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self._sample()
