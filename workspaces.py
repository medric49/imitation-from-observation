from pathlib import Path
import random

import hydra
import numpy as np
import torch
import torch.utils.data
from dm_env import specs
from hydra.utils import to_absolute_path

import ct_model
import datasets
import dmc
import utils
import video
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def _make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = _make_agent(self.train_env.observation_spec(),
                                 self.train_env.action_spec(),
                                 self.cfg.agent)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed, self.cfg.get('xml_path', None))
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, self.cfg.get('xml_path', None))
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


class CTWorkspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)

        self.context_translator: ct_model.CTNet = hydra.utils.instantiate(self.cfg.ct_model).to(utils.device())

        self.dataset = datasets.VideoDataset(to_absolute_path(self.cfg.train_video_dir), self.cfg.episode_len, self.cfg.train_cams, self.cfg.same_video)
        self.valid_dataset = datasets.VideoDataset(to_absolute_path(self.cfg.valid_video_dir), self.cfg.episode_len, self.cfg.train_cams, self.cfg.same_video)

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

        self.context_translator.train()

        while train_until_epoch(self._epoch):
            video1, video2 = next(self.dataloader_iter)
            video1 = video1.to(device=utils.device())
            video2 = video2.to(device=utils.device())
            metrics = self.context_translator.update(video1, video2)
            self.logger.log_metrics(metrics, self._epoch, 'train')

            print(f'E: {self._epoch+1}', end='\t| ')
            for k, v in metrics.items():
                print(f'{k}: {v}', end='\t| ')
            print('')

            if eval_every_epoch(self._epoch):
                self.context_translator.eval()
                with torch.no_grad():
                    eval_loss = 0
                    eval_trans_loss = 0
                    eval_rec_loss = 0
                    eval_align_loss = 0
                    eval_sim_loss = 0
                    for _ in range(self.cfg.num_evaluations):
                        video1, video2 = next(self.valid_dataloader_iter)
                        video1 = video1.to(device=utils.device())
                        video2 = video2.to(device=utils.device())
                        loss, trans_loss, rec_loss, align_loss, sim_loss = self.context_translator.evaluate(video1, video2)

                        eval_loss += loss
                        eval_trans_loss += trans_loss
                        eval_rec_loss += rec_loss
                        eval_align_loss += align_loss
                        eval_sim_loss += sim_loss

                    eval_loss /= self.cfg.num_evaluations
                    eval_trans_loss /= self.cfg.num_evaluations
                    eval_rec_loss /= self.cfg.num_evaluations
                    eval_align_loss /= self.cfg.num_evaluations
                    eval_sim_loss /= self.cfg.num_evaluations
                    metrics = {
                        'loss': eval_loss.item(),
                        'trans_loss': eval_trans_loss.item(),
                        'rec_loss': eval_rec_loss.item(),
                        'align_loss': eval_align_loss.item(),
                        'eval_sim_loss': eval_sim_loss.item()
                    }
                    self.logger.log_metrics(metrics, self._epoch, 'eval')

                    print('Eval loss: ', eval_loss.item(), end='\t')
                    if eval_loss < self._eval_loss:
                        self._eval_loss = eval_loss
                        self.save_snapshot(as_optimal=True)
                        print('*** save ***', end='')
                    print('')

                    video1, video2 = next(self.valid_dataloader_iter)
                    video1 = video1.to(device=utils.device())
                    video2 = video2.to(device=utils.device())
                    video1 = video1[0]  # T x c x h x w
                    fobs2 = video2[0][0]  # c x h x w
                    video2 = self.context_translator.translate(video1, fobs2)
                    video.make_video_from_frames(self.work_dir / f'eval_video/{self._epoch}_expert.mp4', video1.cpu().numpy())
                    video.make_video_from_frames(self.work_dir / f'eval_video/{self._epoch}_agent.mp4', video2.cpu().numpy())
                self.context_translator.train()
            self.save_snapshot()
            self._epoch += 1

    def save_snapshot(self, as_optimal=False):
        if not as_optimal:
            snapshot = self.work_dir / 'snapshot.pt'
        else:
            snapshot = self.work_dir / 'opt_snapshot.pt'
        keys_to_save = ['context_translator', '_epoch', '_eval_loss']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
