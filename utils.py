import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from tqdm import tqdm


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def generate_video_from_expert(root_dir, expert, env, context_changer, cam_ids, num_frames, num_train=800, num_valid=200):
    root_dir = Path(root_dir)
    root_dir.mkdir(exist_ok=True)

    expert.train(training=False)

    im_w, im_h = 64, 64

    def act(time_step):
        action = expert.act(time_step.observation, 1, eval_mode=True)
        return action

    def make_video(parent_dir):
        cameras = {id: [] for id in cam_ids}
        context_changer.reset()

        time_step = env.reset()

        with change_context(env, context_changer):
            for cam_id, cam in cameras.items():
                cam.append(env.physics.render(im_w, im_h, camera_id=cam_id))

        for _ in range(num_frames):
            action = act(time_step)
            time_step = env.step(action)

            with change_context(env, context_changer):
                for cam_id, cam in cameras.items():
                    cam.append(env.physics.render(im_w, im_h, camera_id=cam_id))

        videos = np.array(list(cameras.values()), dtype=np.uint8)
        np.save(parent_dir / f'{int(time.time()*1000)}', videos)

    with torch.no_grad():
        video_dir = root_dir / 'train'
        video_dir.mkdir(exist_ok=True)
        for i in tqdm(range(num_train)):
            make_video(video_dir)
        video_dir = root_dir / 'valid'
        video_dir.mkdir(exist_ok=True)
        for i in tqdm(range(num_valid)):
            make_video(video_dir)


class ReacherHardContextChanger:

    def __init__(self):
        self.reset()

    def reset(self):
        self.ground_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.target_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.arm_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.root_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.finger_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]

    def change_env(self, env):
        env.physics.named.model.mat_texid['grid'] = -1
        env.physics.named.model.geom_rgba['ground'] = self.ground_color
        env.physics.named.model.geom_rgba['target'] = self.target_color
        env.physics.named.model.geom_rgba['arm'] = self.arm_color
        env.physics.named.model.geom_rgba['hand'] = self.arm_color
        env.physics.named.model.geom_rgba['root'] = self.root_color
        env.physics.named.model.geom_rgba['finger'] = self.finger_color

    def reset_env(self, env):
        env.physics.named.model.mat_texid['grid'] = 1
        env.physics.named.model.geom_rgba['ground'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['target'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['arm'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['hand'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['root'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['finger'] = [0.5, 0.5, 0.5, 1]


class change_context:
    def __init__(self, env, context_changer):
        self.env = env
        self.context_changer = context_changer

    def __enter__(self):
        self.context_changer.change_env(self.env)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context_changer.reset_env(self.env)



