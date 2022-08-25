import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch.utils.data
from PIL import Image
from torch.utils.data.dataset import T_co


class CTVideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, episode_len):
        self._root = Path(root)
        self._files = list(self._root.iterdir())
        self._episode_len = episode_len + 1

    def _sample(self):
        videos1, videos2 = random.choices(self._files, k=2)
        video1 = np.load(videos1)[0, :self._episode_len]
        video2 = np.load(videos2)[0, :self._episode_len]
        video1 = video1.transpose(0, 3, 1, 2).copy()
        video2 = video2.transpose(0, 3, 1, 2).copy()
        return video1, video2

    @staticmethod
    def sample_from_dir(video_dir, episode_len=None, im_w=64, im_h=64):
        if episode_len is not None:
            episode_len += 1
        else:
            episode_len = -1
        video_dir = Path(video_dir)
        files = list(video_dir.iterdir())
        video_i = np.load(random.choice(files))[0, :episode_len]
        if tuple(video_i.shape[1:3]) != (im_h, im_w):
            video_i = CTVideoDataset.resize(video_i, im_w, im_h)
        return video_i

    @staticmethod
    def resize(video, im_w, im_h):
        frame_list = []
        for t in range(video.shape[0]):
            frame = Image.fromarray(video[t])
            frame = np.array(frame.resize((im_w, im_h), Image.BICUBIC), dtype=np.float32)
            frame_list.append(frame)
        frame_list = np.stack(frame_list)
        return frame_list

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield self._sample()
