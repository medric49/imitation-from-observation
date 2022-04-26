import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.utils.data
from dm_env import specs
from hydra.utils import to_absolute_path

import context_changers
import ct_model
import dmc
import drqv2
import rl_model
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, ReplayBuffer
from video import VideoRecorder


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

        self.context_translator: ct_model.CTNet = ct_model.CTNet.load(to_absolute_path(self.cfg.ct_file)).to(
            utils.device())
        self.context_translator.eval()

        self.setup()

        self.cfg.agent.action_shape = self.train_env.action_spec().shape
        self.cfg.agent.state_dim = self.train_env.observation_spec().shape[0]
        self.agent: rl_model.ACAgent = hydra.utils.instantiate(self.cfg.agent).to(utils.device())

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        # create envs
        self.expert_env = dmc.make(self.cfg.task_name, self.cfg.expert_frame_stack,
                                   self.cfg.action_repeat, self.cfg.seed, self.cfg.get('xml_path', None),
                                   episode_len=self.cfg.episode_len)

        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed, self.cfg.get('xml_path', None),
                                  self.cfg.learner_camera_id, self.cfg.im_w, self.cfg.im_h,
                                  hydra.utils.instantiate(self.cfg.context_changer),
                                  episode_len=self.cfg.episode_len)
        self.train_env = dmc.EncodeStackWrapper(self.train_env, self.expert, self.context_translator, self.expert_env,
                                                self.cfg.context_camera_ids, self.cfg.n_video, self.cfg.im_w,
                                                self.cfg.im_h, self.cfg.agent.state_dim, self.cfg.frame_stack,
                                                dist_reward=True)

        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, self.cfg.get('xml_path', None),
                                 self.cfg.learner_camera_id, self.cfg.im_w, self.cfg.im_h,
                                 context_changers.ReacherHardContextChanger(),
                                 episode_len=self.cfg.episode_len)
        self.eval_env = dmc.EncodeStackWrapper(self.eval_env, self.expert, self.context_translator, self.expert_env,
                                               self.cfg.context_camera_ids, self.cfg.n_video, self.cfg.im_w,
                                               self.cfg.im_h, self.cfg.agent.state_dim, self.cfg.frame_stack,
                                               dist_reward=False)

        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount')
        )

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_buffer = ReplayBuffer(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.replay_buffer_num_workers, self.cfg.nstep,
            self.cfg.discount, 1, self.cfg.save_snapshot
        )

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)

    def collect_steps(self, init_step):
        seed_until_step = utils.Until(self.cfg.batch_size)

        nb_steps = 0

        while seed_until_step(nb_steps):
            episode_step = 0

            time_step = self.train_env.reset()
            self.replay_storage.add(time_step)

            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    state = torch.tensor(time_step.observation, device=utils.device(), dtype=torch.float)
                    action = self.agent.act(state, init_step, eval_mode=False)

                time_step = self.train_env.step(action)
                self.replay_storage.add(time_step)

            nb_steps += episode_step

    def train(self):
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        while train_until_step(self.global_step):
            init_step = self.global_step
            print('Collect data...')
            self.collect_steps(init_step)
            print('Train...')
            metrics = self.agent.update(self.replay_buffer, self.cfg.batch_size, self.cfg.nstep, init_step)
            self.logger.log_metrics(metrics, self.global_frame, ty='train')
            print('Eval...')
            self.eval()
            self.save_snapshot()

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            episode_step = 0
            time_step = self.eval_env.reset()

            self.video_recorder.init(self.eval_env)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    state = torch.tensor(time_step.observation, device=utils.device(), dtype=torch.float)
                    action = self.agent.act(state, self.global_step, eval_mode=True)

                time_step = self.eval_env.step(action)
                episode_step += 1

                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}_{episode}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

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


@hydra.main(config_path='ac_cfgs', config_name='config')
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
