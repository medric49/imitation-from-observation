import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch.utils.data
from torch import nn
from torch.utils.data.dataset import T_co
from torch.nn import functional as F
from PIL import Image
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


class VideoDataset(torch.utils.data.IterableDataset):

    def __init__(self, root, episode_len, cam_ids, to_lab=False, im_w=64, im_h=64):
        self._root = Path(root)
        self._num_classes = len(list(self._root.iterdir()))

        self.im_w = im_w
        self.im_h = im_h
        self.update_files()

        self._episode_len = episode_len + 1
        self._cam_ids = cam_ids
        self.to_lab = to_lab

    def update_files(self, max_num_video=None):
        self._files = []
        for c in range(self._num_classes):
            class_dir = self._root / str(c)
            files = list(sorted(class_dir.iterdir()))
            if max_num_video is not None and len(files) > max_num_video:
                old_files = files[:-15000]
                files = files[-15000:]
                for f in old_files:
                    os.remove(f)
            self._files.append(files)



    def _sample(self):
        if len(self._cam_ids) > 1:
            cam1, cam2 = random.sample(self._cam_ids, k=3)
        else:
            cam1, cam2 = 0, 0

        classes = list(range(self._num_classes))
        class_1 = 0
        classes.remove(class_1)
        class_2 = random.choice(classes)

        video_i = random.choice(self._files[class_1])
        video_n = random.choice(self._files[class_2])

        video_i = np.load(video_i)[cam1, :self._episode_len]
        video_n = np.load(video_n)[cam2, :self._episode_len]

        if tuple(video_i.shape[1:3]) != (self.im_h, self.im_w):
            video_i = VideoDataset.resize(video_i, self.im_w, self.im_h)
        if tuple(video_n.shape[1:3]) != (self.im_h, self.im_w):
            video_n = VideoDataset.resize(video_n, self.im_w, self.im_h)

        if self.to_lab:
            video_i = VideoDataset.rgb_to_lab(video_i)
            video_n = VideoDataset.rgb_to_lab(video_n)
        else:
            video_i /= 255.
            video_n /= 255.
            video_i = utils.normalize(video_i, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            video_n = utils.normalize(video_n, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        video_i = video_i.transpose(0, 3, 1, 2).copy()
        video_n = video_n.transpose(0, 3, 1, 2).copy()

        return video_i, video_n

    @staticmethod
    def resize(video, im_w, im_h):
        frame_list = []
        for t in range(video.shape[0]):
            frame = Image.fromarray(video[t])
            frame = np.array(frame.resize((im_w, im_h), Image.BICUBIC), dtype=np.float32)
            frame_list.append(frame)
        frame_list = np.stack(frame_list)
        return frame_list

    @staticmethod
    def rgb_to_lab(video):
        T = video.shape[0]
        return np.array([utils.rgb_to_lab(video[t]) for t in range(T)])

    @staticmethod
    def sample_from_dir(video_dir, episode_len=None, im_w=64, im_h=64, to_lab=False):
        if episode_len is not None:
            episode_len += 1
        else:
            episode_len = -1

        video_dir = Path(video_dir)
        files = list(video_dir.iterdir())
        video_i = np.load(random.choice(files))[0, :episode_len]
        if tuple(video_i.shape[1:3]) != (im_h, im_w):
            video_i = VideoDataset.resize(video_i, im_w, im_h)
        if to_lab:
            video_i = VideoDataset.rgb_to_lab(video_i)
        else:
            video_i /= 255.
            video_i = utils.normalize(video_i, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return video_i

    @staticmethod
    def augment(video_i: torch.Tensor, video_n: torch.Tensor):
        T = video_i.shape[1]
        p_list = [0.05 for i in range(T)]
        indices = [i for i in range(T) if np.random.rand() > p_list[i]]
        video_i = video_i[:, indices, :, :, :]
        video_n = video_n[:, indices, :, :, :]
        return video_i, video_n

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self._sample()
