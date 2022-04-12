import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Encoder(nn.Module):
    def __init__(self, state_dim, repr_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, repr_dim),
            nn.BatchNorm1d(repr_dim),
            nn.LeakyReLU(),
            nn.Linear(repr_dim, repr_dim),
        )
        self.apply(utils.weight_init)

    def forward(self, state):
        state = self.encoder(state)
        return state


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, state, std):
        h = self.trunk(state)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, state, action):
        h = self.trunk(state)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class RLAgent(nn.Module):
    def __init__(self, state_dim, repr_dim, action_shape, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        super(RLAgent, self).__init__()

        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = Encoder(state_dim, repr_dim)
        self.actor = Actor(repr_dim, action_shape, feature_dim, hidden_dim)
        self.critic = Critic(repr_dim, action_shape, feature_dim, hidden_dim)
        self.critic_target = Critic(repr_dim, action_shape, feature_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)


        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, state, step, eval_mode):
        state = state.unsqueeze(0)
        state = self.encoder.encoder(state)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(state, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, state, action, reward, discount, next_state, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_state, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_V = torch.min(target_Q1, target_Q2)

            # target_V = utils.normalize(target_V, target_V.mean(), target_V.std())
            # reward = utils.normalize(reward, reward.mean(), reward.std())

            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, state, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(state, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(state, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

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

        state = self.encoder(state)
        with torch.no_grad():
            next_state = self.encoder(next_state)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(state, action, reward, discount, next_state, step))

        # update actor
        metrics.update(self.update_actor(state.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            payload = torch.load(f)
        return payload['agent']


class ACAgent(nn.Module):
    def __init__(self, state_dim, repr_dim, action_shape, feature_dim, hidden_dim, lr, stddev_schedule, stddev_clip, critic_target_tau, with_target_critic):
        super(ACAgent, self).__init__()

        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.critic_target_tau = critic_target_tau
        self.with_target_critic = with_target_critic

        self.encoder = Encoder(state_dim, repr_dim)
        self.actor = Actor(repr_dim, action_shape, feature_dim, hidden_dim)
        self.critic = Critic(repr_dim, action_shape, feature_dim, hidden_dim)

        if self.with_target_critic:
            self.critic_target = Critic(repr_dim, action_shape, feature_dim, hidden_dim)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_target.train()

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr)

        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def eval(self):
        self.train(False)

    def act(self, state, step, eval_mode):
        state = state.unsqueeze(0)
        state = self.encoder.encoder(state)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(state, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
        return action.cpu().numpy()[0]

    def update_critic(self, state, action, reward, discount, next_state, terminal, step):
        metrics = dict()

        with torch.no_grad():
            next_state = self.encoder(next_state)
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_state, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            if self.with_target_critic:
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            else:
                target_Q1, target_Q2 = self.critic(next_state, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V * (1 - terminal))

        state = self.encoder(state)
        Q1, Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, state, action, advantage, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(state, stddev)

        log_prob = dist.log_prob(action)
        actor_loss = - (log_prob * advantage).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_logprob'] = log_prob.sum(-1, keepdim=True).mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_buffer, batch_size, nstep, step):
        metrics = dict()

        batch = replay_buffer.sample_recent_data(batch_size, nstep)
        state, action, reward, discount, next_state, terminal = utils.to_torch(
            batch, utils.device())
        metrics['batch_reward'] = reward.mean().item()

        critic_metrics = None
        for _ in range(3):
            critic_metrics = self.update_critic(state, action, reward, discount, next_state, terminal, step)
        metrics.update(critic_metrics)

        with torch.no_grad():
            next_state = self.encoder(next_state)
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_state, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            q1, q2 = self.critic(next_state, next_action)
            next_value = torch.min(q1, q2)

            state = self.encoder(state)
            q1, q2 = self.critic(state, action)
            value = torch.min(q1, q2)

            advantage = value - next_value

            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_metrics = None
        for _ in range(3):
            # update actor
            actor_metrics = self.update_actor(state, action, advantage, step)
        metrics.update(actor_metrics)

        # update critic target
        if self.with_target_critic:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)

        return metrics
