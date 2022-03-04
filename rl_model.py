import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim * 2, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, state, target_state, std):
        current_h = self.trunk(state)
        target_h = self.trunk(target_state)
        h = torch.cat([current_h, target_h], dim=1)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Dynamics(nn.Module):
    def __init__(self, repr_dim, action_shape):
        super().__init__()

        self.p = nn.Sequential(
            nn.Linear(repr_dim + action_shape[0], 512), nn.LeakyReLU(),
            nn.Linear(512, 128), nn.LeakyReLU(),
            nn.Linear(128, 128), nn.LeakyReLU(),
            nn.Linear(128, 128), nn.LeakyReLU(),
            nn.Linear(128, 512), nn.LeakyReLU(),
            nn.Linear(512, repr_dim)
        )

        self.apply(utils.weight_init)

    def forward(self, state, action):
        h_action = torch.cat([state, action], dim=-1)
        pred_state = self.p(h_action)

        return pred_state


class RLAgent(nn.Module):
    def __init__(self, state_dim, repr_dim, action_shape, lr, feature_dim,
                 hidden_dim, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        super(RLAgent, self).__init__()

        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.state_dim = state_dim

        # models
        self.actor = Actor(state_dim, action_shape, feature_dim, hidden_dim)
        self.dynamics = Dynamics(state_dim, action_shape)

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.dynamics_opt = torch.optim.Adam(self.dynamics.parameters(), lr=lr)

        # data augmentation

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.dynamics.train(training)

    def act(self, state, step, eval_mode):
        state = state.unsqueeze(0)
        state, target_state = state[:, :self.state_dim], state[:, self.state_dim:]

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(state, target_state, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_dynamics(self, state, action, next_state):
        metrics = dict()

        pred_state = self.dynamics(state, action)
        dynamics_loss = F.mse_loss(pred_state, next_state)

        if self.use_tb:
            metrics['dynamics_loss'] = dynamics_loss.item()

        # optimize dynamics
        self.dynamics_opt.zero_grad(set_to_none=True)
        dynamics_loss.backward()
        self.dynamics_opt.step()

        return metrics

    def update_actor(self, state, target_state, step):
        metrics = dict()

        self.dynamics.eval()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(state, target_state, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        pred_state = self.dynamics(state, action)

        actor_loss = F.mse_loss(pred_state, target_state)

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        self.dynamics.train()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        state, action, reward, discount, next_state = utils.to_torch(
            batch, utils.device())
        state, target_state = state[:, :self.state_dim], state[:, self.state_dim:]
        next_state = next_state[:, :self.state_dim]

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update dynamics
        metrics.update(
            self.update_dynamics(state, action, next_state))

        # update actor
        metrics.update(self.update_actor(state, target_state, step))

        return metrics

    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            payload = torch.load(f)
        return payload['agent']
