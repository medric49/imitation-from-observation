import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch.utils.data
from torch import nn
from torch.utils.data.dataset import T_co
from torch.nn import functional as F

import utils


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

    def __init__(self, root, episode_len, cam_ids, to_lab=False):
        self._root = Path(root)
        self._num_classes = len(list(self._root.iterdir()))
        self._files = []

        for c in range(self._num_classes):
            class_dir = self._root / str(c)
            self._files.append(list(class_dir.iterdir()))

        self._episode_len = episode_len
        self._cam_ids = cam_ids
        self.to_lab = to_lab

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

        if self.to_lab:
            video_i = self.rgb_to_lab(video_i)
            video_p = self.rgb_to_lab(video_p)
            video_n = self.rgb_to_lab(video_n)

        video_i = video_i.transpose(0, 3, 1, 2).copy()
        video_p = video_p.transpose(0, 3, 1, 2).copy()
        video_n = video_n.transpose(0, 3, 1, 2).copy()

        return video_i, video_p, video_n

    def rgb_to_lab(self, video):
        T = video.shape[0]
        return np.array([utils.rgb_to_lab(video[t]) for t in range(T)])


    @staticmethod
    def augment(video_i: torch.Tensor, video_p: torch.Tensor, video_n: torch.Tensor):
        # video_i, video_p, video_n = ViRLVideoDataset.augment_images(video_i, video_p, video_n)
        # video_i, video_p, video_n = ViRLVideoDataset.add_noise(video_i, video_p, video_n)
        # video_i, video_p, video_n = ViRLVideoDataset.random_shuffle(video_i, video_p, video_n)
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
        T = video_i.shape[1]
        # base = sum(list(range(T)))
        # p_list = [(T - i)/base for i in range(T)]
        p_list = [1./10 for i in range(T)]

        indices = [i for i in range(T) if np.random.rand() > p_list[i]]
        video_1 = video_i[:, indices, :, :, :]
        video_2 = video_p[:, indices, :, :, :]
        video_3 = video_n[:, indices, :, :, :]

        return video_1, video_2, video_3

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self._sample()


class CMCVideoDataset(torch.utils.data.IterableDataset):

    def __init__(self, root, episode_len, cam_ids, batch_size, to_lab=False):
        self._root = Path(root)
        self._num_classes = len(list(self._root.iterdir()))
        self._files = []

        for c in range(self._num_classes):
            class_dir = self._root / str(c)
            self._files.append(list(class_dir.iterdir()))

        self._episode_len = episode_len
        self._cam_ids = cam_ids
        self.to_lab = to_lab

        self.batch_size = batch_size
        self.batch_item = 0

        self.class_1 = None
        self.class_2 = None

    def _sample(self):
        if len(self._cam_ids) > 1:
            cam1, cam2, cam3 = random.sample(self._cam_ids, k=3)
        else:
            cam1, cam2, cam3 = 0, 0, 0

        if self.batch_item == 0:
            classes = list(range(self._num_classes))
            self.class_1 = random.choice(classes)
            classes.remove(self.class_1)
            self.class_2 = random.choice(classes)

        if self.batch_item in [0, 1]:
            item_class = self.class_1
        else:
            item_class = self.class_2

        video = random.choice(self._files[item_class])
        video = np.load(video)[cam1, :self._episode_len]

        if self.to_lab:
            video = self.rgb_to_lab(video)

        video = video.transpose(0, 3, 1, 2).copy()

        self.batch_item = (self.batch_item + 1) % self.batch_size

        return video, item_class

    def rgb_to_lab(self, video):
        T = video.shape[0]
        return np.array([utils.rgb_to_lab(video[t]) for t in range(T)])

    @staticmethod
    def augment(video: torch.Tensor):
        video = ViRLVideoDataset.random_sequence_cropping(video)
        return video

    @staticmethod
    def random_sequence_cropping(video: torch.Tensor):
        T = video.shape[1]
        # base = sum(list(range(T)))
        # p_list = [(T - i)/base for i in range(T)]
        p_list = [1./10 for i in range(T)]

        indices = [i for i in range(T) if np.random.rand() > p_list[i]]
        video = video[:, indices, :, :, :]

        return video

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self._sample()
