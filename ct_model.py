import random
from pathlib import Path

import gc

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CTNet(nn.Module):
    def __init__(self, hidden_dim, lr, use_tb):
        super(CTNet, self).__init__()

        self.use_tb = use_tb

        self.enc1 = EncoderNet(hidden_dim)
        self.enc2 = EncoderNet(hidden_dim)
        self.t = LSTMTranslatorNet(hidden_dim)
        self.dec = DecoderNet(hidden_dim)

        self._enc1_opt = torch.optim.Adam(self.enc1.parameters(), lr=lr)
        self._enc2_opt = torch.optim.Adam(self.enc2.parameters(), lr=lr)
        self._t_opt = torch.optim.Adam(self.t.parameters(), lr=lr)
        self._dec_opt = torch.optim.Adam(self.dec.parameters(), lr=lr)
    
    def translate(self, video1, fobs2, keep_enc2=True):
        T = video1.shape[0]

        video1 = video1.to(dtype=torch.float) / 255.  # T x c x h x w
        fobs2 = fobs2.to(dtype=torch.float) / 255.  # c x h x w
        video1 = video1.unsqueeze(dim=1)
        fobs2 = fobs2.unsqueeze(dim=0)

        z1_seq = [self.enc1(video1[t]) for t in range(T)]
        z1_seq = torch.stack(z1_seq)

        fz2 = self.enc2(fobs2)

        z3_seq = self.t(z1_seq, fz2)

        video2 = [self.dec(z3_seq[t]) for t in range(T)]  # T x c x h x w
        video2 = torch.stack(video2)

        z3_seq = z3_seq.squeeze(dim=1)
        video2 = video2.squeeze(dim=1)
        if keep_enc2:
            video2[0] = fobs2[0]
        video2 = video2 * 255.
        video2[video2 > 255.] = 255.
        video2[video2 < 0.] = 0.
        return z3_seq, video2

    def evaluate(self, video1, video2):
        T = video1.shape[1]
        n = video1.shape[0]

        video1 = video1.to(dtype=torch.float) / 255.  # n x T x c x h x w
        video2 = video2.to(dtype=torch.float) / 255.  # n x T x c x h x w

        video1 = torch.transpose(video1, dim0=0, dim1=1)  # T x n x c x h x w
        video2 = torch.transpose(video2, dim0=0, dim1=1)  # T x n x c x h x w

        fobs2 = video2[0]  # n x c x h x w

        l_trans = 0
        l_rec = 0
        l_align = 0

        fz2 = self.enc2(fobs2)

        z1_seq = [self.enc1(video1[t]) for t in range(T)]
        z1_seq = torch.stack(z1_seq)
        z3_seq = self.t(z1_seq, fz2)

        z2_seq = [self.enc1(video2[t]) for t in range(T)]
        z2_seq = torch.stack(z2_seq)

        for t in range(T):
            obs2 = video2[t]
            obs_z3 = self.dec(z3_seq[t])
            obs_z2 = self.dec(z2_seq[t])

            l_trans += F.mse_loss(torch.flatten(obs_z3, start_dim=1), torch.flatten(obs2, start_dim=1))
            l_rec += F.mse_loss(torch.flatten(obs_z2, start_dim=1), torch.flatten(obs2, start_dim=1))
            l_align += F.mse_loss(z3_seq[t], z2_seq[t])

        loss = 0.
        loss += l_trans * 1.
        loss += l_rec * 1.
        loss += l_align * 1.

        metrics = {
            'loss': loss.item(),
            'l_trans': l_trans.item(),
            'l_rec': l_rec.item(),
            'l_align': l_align.item(),
        }

        return metrics, loss

    def update(self, video1, video2):
        self._enc1_opt.zero_grad()
        self._enc2_opt.zero_grad()
        self._t_opt.zero_grad()
        self._dec_opt.zero_grad()

        metrics, loss = self.evaluate(video1, video2)
        
        loss.backward()
        
        self._dec_opt.step()
        self._t_opt.step()
        self._enc2_opt.step()
        self._enc1_opt.step()

        return metrics

    def encode(self, obs):
        obs = obs.to(dtype=torch.float) / 255.
        obs = self.enc1(obs)
        return obs

    @staticmethod
    def load(file):
        snapshot = Path(file)
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        return payload['context_translator']

            
class EncoderNet(nn.Module):
    def __init__(self, hidden_dim):
        super(EncoderNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Flatten()
        )

    def forward(self, obs):
        return self.net(obs)


class DecoderNet(nn.Module):
    def __init__(self, hidden_dim):
        super(DecoderNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, output_padding=1)
        )

    def forward(self, z):
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        return self.net(z)


class LSTMTranslatorNet(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTMTranslatorNet, self).__init__()
        self.num_layers = 2
        self.translator = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=self.num_layers)

    def forward(self, z1_seq, z2):
        c0 = z2.repeat(self.num_layers, 1, 1)
        h0 = torch.zeros_like(z2).repeat(self.num_layers, 1, 1)
        z3_seq, _ = self.translator(z1_seq, (h0, c0))
        return z3_seq
