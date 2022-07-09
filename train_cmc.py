import sys
import warnings

import cmc_model
import virl_model

warnings.filterwarnings('ignore', category=DeprecationWarning)

import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.utils.data
from hydra.utils import to_absolute_path

import datasets
import utils
from logger import Logger


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def _make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg, hyperparams_str=''):
        self.work_dir = Path.cwd() / hyperparams_str
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.encoder: cmc_model.CMCModel = hydra.utils.instantiate(self.cfg.cmc_model).to(utils.device())

        self.dataset = datasets.CMCVideoDataset(to_absolute_path(self.cfg.train_video_dir), self.cfg.episode_len, self.cfg.train_cams, self.cfg.batch_size, to_lab=True)
        self.valid_dataset = datasets.CMCVideoDataset(to_absolute_path(self.cfg.valid_video_dir), self.cfg.episode_len, self.cfg.train_cams, self.cfg.batch_size, to_lab=True)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            worker_init_fn=_worker_init_fn,
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.batch_size
        )

        self.dataloader_iter = iter(self.dataloader)
        self.valid_dataloader_iter = iter(self.valid_dataloader)

        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        self._epoch = 0
        self._eval_loss = np.inf

    def train(self):
        train_until_epoch = utils.Until(self.cfg.num_epochs)
        eval_every_epoch = utils.Every(self.cfg.eval_every_epochs)

        self.encoder.train()

        while train_until_epoch(self._epoch):
            video = next(self.dataloader_iter)
            video = video.to(device=utils.device(), dtype=torch.float)
            video = datasets.CMCVideoDataset.augment(video)

            metrics = self.encoder.update(video)

            self.logger.log_metrics(metrics, self._epoch, 'train')

            print(f'E: {self._epoch+1}', end='\t| ')
            for k, v in metrics.items():
                print(f'{k}: {v}', end='\t| ')
            print('')

            if eval_every_epoch(self._epoch):
                self.encoder.eval()
                with torch.no_grad():
                    metrics = None
                    for _ in range(self.cfg.num_evaluations):
                        video = next(self.valid_dataloader_iter)
                        video = video.to(device=utils.device(), dtype=torch.float)
                        m, _ = self.encoder.evaluate(video)

                        if metrics is None:
                            metrics = m
                        else:
                            for k, v in m.items():
                                metrics[k] += v

                    for k, v in metrics.items():
                        metrics[k] /= self.cfg.num_evaluations

                    self.logger.log_metrics(metrics, self._epoch, 'eval')

                    eval_loss = metrics['loss']
                    print('Eval loss: ', eval_loss, end='\t')
                    if eval_loss < self._eval_loss:
                        self._eval_loss = eval_loss
                        self.save_snapshot(as_optimal=True)
                        print('*** save ***', end='')
                    print('')

                self.encoder.train()

            self.save_snapshot()
            self._epoch += 1

    def save_snapshot(self, as_optimal=False):
        if not as_optimal:
            snapshot = self.work_dir / 'snapshot.pt'
        else:
            snapshot = self.work_dir / 'opt_snapshot.pt'
        keys_to_save = ['encoder', '_epoch', '_eval_loss']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cmc_cfgs', config_name='config')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()