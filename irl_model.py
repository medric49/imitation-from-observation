import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class InvDynamics(nn.Module):
    def __init__(self, repr_dim, action_shape):
        super().__init__()

        self.inv_p = nn.Sequential(
            nn.Linear(repr_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, action_shape[0]),
            nn.Tanh()
        )
        self.apply(utils.weight_init)

    def forward(self, state, next_state):
        curr_next_states = torch.cat([state, next_state], dim=1)
        action = self.inv_p(curr_next_states)
        return action


class IRLAgent(nn.Module):
    def __init__(self, state_dim, repr_dim, action_shape, lr, use_tb):

        super(IRLAgent, self).__init__()
        self.use_tb = use_tb

        self.state_dim = state_dim

        # models
        self.inv_dynamics = InvDynamics(state_dim, action_shape)

        # optimizers
        self.inv_dynamics_opt = torch.optim.Adam(self.inv_dynamics.parameters(), lr=lr)

        # data augmentation

        self.train()

    def update(self, replay_iter):
        batch = next(replay_iter)
        state, action, reward, discount, next_state = utils.to_torch(batch, utils.device())

        pred_action = self.inv_dynamics(state, next_state)
        loss = F.mse_loss(pred_action, action)

        self.inv_dynamics_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.inv_dynamics_opt.step()

        return loss.item()

    def act(self, state, target_state):
        state = state.unsqueeze(0)
        target_state = target_state.unsqueeze(0)
        action = self.inv_dynamics(state, target_state)[0]
        return action
