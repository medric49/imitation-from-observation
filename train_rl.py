import os
import shutil
import warnings
import time

import cmc_model
import datasets
import metaworld_env
import virl_model

warnings.filterwarnings('ignore', category=DeprecationWarning)

import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.utils.data
from dm_env import specs
from hydra.utils import to_absolute_path

import dmc
import drqv2
import rl_model
import utils
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

        self.expert: drqv2.DrQV2Agent = drqv2.DrQV2Agent.load(to_absolute_path(self.cfg.expert_file))
        self.expert.train(training=False)

        video_dir = Path(to_absolute_path(self.cfg.video_dir))
        if video_dir.exists():
            shutil.rmtree(video_dir)
        Path(video_dir / '0').mkdir(exist_ok=True, parents=True)
        Path(video_dir / '1').mkdir(exist_ok=True, parents=True)

        self.setup()

        self.dataset = datasets.ViRLVideoDataset(to_absolute_path(self.cfg.video_dir), self.cfg.episode_len,
                                                 self.cfg.train_cams, to_lab=self.cfg.to_lab)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.cfg.enc_batch_size,
        )
        self.dataloader_iter = iter(self.dataloader)

        self.cfg.agent.action_shape = self.train_env.action_spec().shape
        self.cfg.agent.state_dim = self.train_env.observation_spec().shape[0]
        self.rl_agent: rl_model.RLAgent = hydra.utils.instantiate(self.cfg.agent).to(utils.device())

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        # create envs
        if not self.cfg.get('meta_world', None):
            self.expert_env = dmc.make(self.cfg.task_name, self.cfg.expert_frame_stack,
                                       self.cfg.action_repeat, self.cfg.seed, self.cfg.get('xml_path', None),
                                       episode_len=self.cfg.episode_len)
            self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                      self.cfg.action_repeat, self.cfg.seed, self.cfg.get('xml_path', None),
                                      self.cfg.learner_camera_id, self.cfg.im_w, self.cfg.im_h,
                                      hydra.utils.instantiate(self.cfg.context_changer),
                                      episode_len=self.cfg.episode_len, to_lab=self.cfg.to_lab, normalize_img=not self.cfg.to_lab)

            self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                     self.cfg.action_repeat, self.cfg.seed, self.cfg.get('xml_path', None),
                                     self.cfg.learner_camera_id, self.cfg.im_w, self.cfg.im_h,
                                     hydra.utils.instantiate(self.cfg.context_changer),
                                     episode_len=self.cfg.episode_len, to_lab=self.cfg.to_lab, normalize_img=not self.cfg.to_lab)
        else:
            self.expert_env = metaworld_env.Env(self.cfg.task_name)
            self.expert_env = dmc.wrap(self.expert_env, self.cfg.expert_frame_stack, self.cfg.action_repeat, episode_len=self.cfg.episode_len)

            self.train_env = metaworld_env.Env(self.cfg.task_name, self.cfg.im_w, self.cfg.im_h)
            self.train_env = dmc.wrap(self.train_env, self.cfg.frame_stack, self.cfg.action_repeat,
                                      episode_len=self.cfg.episode_len, to_lab=self.cfg.to_lab, normalize_img=not self.cfg.to_lab)

            self.eval_env = metaworld_env.Env(self.cfg.task_name, self.cfg.im_w, self.cfg.im_h)
            self.eval_env = dmc.wrap(self.eval_env, self.cfg.frame_stack, self.cfg.action_repeat,
                                     episode_len=self.cfg.episode_len, to_lab=self.cfg.to_lab, normalize_img=not self.cfg.to_lab)

        self.encoder: cmc_model.CMCModel = cmc_model.CMCModel.load(to_absolute_path(self.cfg.cmc_file)).to(
            utils.device())
        self.encoder.eval()
        self.encoder.deactivate_state_update()
        self.encoder_target: cmc_model.CMCModel = cmc_model.CMCModel.load(to_absolute_path(self.cfg.cmc_file)).to(
            utils.device())
        self.encoder_target.eval()
        for param in self.encoder_target.parameters():
            param.requires_grad = False

        self.train_env = dmc.ViRLEncoderStackWrapper(self.train_env, self.expert, self.encoder_target,
                                                     self.expert_env,
                                                     self.cfg.train_cams, self.cfg.im_w,
                                                     self.cfg.im_h, self.cfg.agent.state_dim, self.cfg.frame_stack,
                                                     hydra.utils.instantiate(self.cfg.context_changer),
                                                     dist_reward=True, to_lab=self.cfg.to_lab)
        self.eval_env = dmc.ViRLEncoderStackWrapper(self.eval_env, self.expert, self.encoder_target,
                                                    self.expert_env,
                                                    self.cfg.train_cams, self.cfg.im_w,
                                                    self.cfg.im_h, self.cfg.agent.state_dim, self.cfg.frame_stack,
                                                    hydra.utils.instantiate(self.cfg.context_changer),
                                                    dist_reward=True, to_lab=self.cfg.to_lab)

        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount')
        )

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, fetch_every=self.cfg.episode_len)
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
            episode_reward = 0.
            self.video_recorder.init(self.eval_env)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.rl_agent):
                    state = torch.tensor(time_step.observation, device=utils.device(), dtype=torch.float)
                    action = self.rl_agent.act(state, self.global_step, eval_mode=True)

                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                episode_reward += time_step.reward
                step += 1

            episode += 1
            total_reward += episode_reward
            self.video_recorder.save(f'{self.global_frame}_{episode}_{episode_reward}.mp4')

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
        train_encoder_every_step = utils.Every(self.cfg.train_encoder_every_steps, self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        frame_sequence = [[]]
        time_step = self.train_env.reset()
        np.save(to_absolute_path(f'{self.cfg.video_dir}/0/{int(time.time() * 1000)}'), self.train_env.expert_frames)
        self.replay_storage.add(time_step)

        frame = self.train_env.physics.render(self.cfg.im_w, self.cfg.im_h, camera_id=self.cfg.learner_camera_id)
        frame_sequence[0].append(frame)
        self.train_video_recorder.init(frame.transpose((2, 0, 1)))

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
                np.save(to_absolute_path(f'{self.cfg.video_dir}/0/{int(time.time() * 1000)}'),
                        self.train_env.expert_frames)
                frame_sequence = np.array(frame_sequence, dtype=np.uint8)
                np.save(to_absolute_path(f'{self.cfg.video_dir}/1/{int(time.time() * 1000)}'), frame_sequence)
                frame_sequence = [[]]
                self.dataset.update_files()

                frame = self.train_env.physics.render(self.cfg.im_w, self.cfg.im_h, camera_id=self.cfg.learner_camera_id)
                frame_sequence[0].append(frame)
                self.train_video_recorder.init(frame.transpose((2, 0, 1)))
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
            with torch.no_grad(), utils.eval_mode(self.rl_agent):
                state = torch.tensor(time_step.observation, device=utils.device(), dtype=torch.float)
                action = self.rl_agent.act(state, self.global_step, eval_mode=False)

            # try to update the encoder
            if not seed_until_step(self.global_step) and train_encoder_every_step(self.global_step):
                video_i, video_n = next(self.dataloader_iter)
                video_i = video_i.to(device=utils.device(), dtype=torch.float)
                video_n = video_n.to(device=utils.device(), dtype=torch.float)
                video_i, video_n = datasets.ViRLVideoDataset.augment(video_i, video_n)
                enc_metrics = self.encoder.update(video_i, video_n)
                utils.soft_update_params(self.encoder, self.encoder_target, self.cfg.critic_target_tau)
                self.logger.log_metrics(enc_metrics, self.global_frame, ty='train')

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.rl_agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            self.replay_storage.add(time_step)

            frame = self.train_env.physics.render(self.cfg.im_w, self.cfg.im_h, camera_id=self.cfg.learner_camera_id)
            frame_sequence[0].append(frame)
            self.train_video_recorder.record(frame.transpose((2, 0, 1)))

            episode_reward += time_step.reward
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['rl_agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='rl_cfgs', config_name='config')
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
