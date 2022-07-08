import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from losses import SupConLoss


class ConvNet(nn.Module):
    def __init__(self, in_channel, hidden_dim):
        super(ConvNet, self).__init__()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv_1 = nn.Conv2d(in_channel, 64, kernel_size=5, stride=2)
        self.b_norm_1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.b_norm_2 = nn.BatchNorm2d(128)
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.b_norm_3 = nn.BatchNorm2d(256)
        self.conv_4 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.b_norm_4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Conv2d(512, hidden_dim * 4, kernel_size=1)
        self.b_norm_fc_1 = nn.BatchNorm2d(hidden_dim * 4)
        self.fc2 = nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1)

    def forward(self, obs):
        c1 = self.leaky_relu(self.b_norm_1(self.conv_1(obs)))
        c2 = self.leaky_relu(self.b_norm_2(self.conv_2(c1)))
        c3 = self.leaky_relu(self.b_norm_3(self.conv_3(c2)))
        c4 = self.leaky_relu(self.b_norm_4(self.conv_4(c3)))
        e = self.leaky_relu(self.b_norm_fc_1(self.fc1(c4)))
        e = self.sigmoid(self.fc2(e))
        e = e.view(e.shape[0], e.shape[1])
        return e, c1, c2, c3, c4


class ConvNet(nn.Module):
    def __init__(self, hidden_dim):
        super(ConvNet, self).__init__()

        self.enc_l = HalfAlexNet(1, hidden_dim // 2)
        self.enc_ab = HalfAlexNet(2, hidden_dim // 2)

    def forward(self, x_l, x_ab):
        x_l = self.enc_l(x_l)
        x_ab = self.enc_ab(x_ab)

        return x_l, x_ab


class CMCModel(nn.Module):

    def __init__(self, hidden_dim, rho, lr):
        super(CMCModel, self).__init__()

        self.rho = rho
        self.hidden_dim = hidden_dim
        self.conv = ConvNet(hidden_dim)
        self.lstm_enc = LSTMEncoder(hidden_dim)

        self.conv_opt = torch.optim.Adam(self.conv.parameters(), lr)
        self.lstm_enc_opt = torch.optim.Adam(self.lstm_enc.parameters(), lr)

        self.contrast_loss = SupConLoss()

    def encode(self, video):
        e_seq = self.encode_frame(video)
        h, _ = self.lstm_enc(e_seq)
        return h[-1]

    def encode_state_seq(self, e_seq):
        h, _ = self.lstm_enc(e_seq)
        return h[-1]

    def encode_frame(self, image):
        shape = image.shape
        if len(shape) == 3:
            image = image.unsqueeze(0)  # 1 x c x h x w
        view_1, view_2 = torch.split(image, [1, 2], dim=1)
        e_1, e_2 = self.conv(view_1, view_2)
        e = torch.cat([e_1, e_2], dim=1)
        if len(shape) == 3:
            e = e.squeeze()
        return e

    def _encode(self, video):
        shape = video.shape  # T x n x c x h x w
        e_1_seq = []
        e_2_seq = []
        for t in range(shape[0]):
            frame = video[t]  # n x c x h x w
            view_1, view_2 = torch.split(frame, [1, 2], dim=1)
            e_1, e_2 = self.conv(view_1, view_2)
            e_1_seq.append(e_1)
            e_2_seq.append(e_2)
        e_1_seq = torch.stack(e_1_seq)  # T x n x z/2
        e_2_seq = torch.stack(e_2_seq)  # T x n x z/2
        return e_1_seq, e_2_seq

    def evaluate(self, video_i, video_p, video_n):
        T = video_i.shape[1]
        n = video_i.shape[0]

        video_i = torch.transpose(video_i, dim0=0, dim1=1)  # T x n x c x h x w
        video_p = torch.transpose(video_p, dim0=0, dim1=1)  # T x n x c x h x w
        video_n = torch.transpose(video_n, dim0=0, dim1=1)  # T x n x c x h x w

        e_i1, e_i2 = self._encode(video_i)  # T x n x z/2
        e_p1, e_p2 = self._encode(video_p)  # T x n x z/2
        e_n1, e_n2 = self._encode(video_n)  # T x n x z/2

        e_1 = torch.cat([e_i1, e_p1, e_n1], dim=1).view(-1, self.hidden_dim)  # 3Tn x z/2
        e_2 = torch.cat([e_i2, e_p2, e_n2], dim=1).view(-1, self.hidden_dim)  # 3Tn x z/2

        e_i_seq = torch.cat([e_i1, e_i2], dim=2)  # T x n x z
        e_p_seq = torch.cat([e_p1, e_p2], dim=2)  # T x n x z
        e_n_seq = torch.cat([e_n1, e_n2], dim=2)  # T x n x z

        h_i_seq, hidden_i = self.lstm_enc(e_i_seq)  # T x n x z
        h_p_seq, hidden_p = self.lstm_enc(e_p_seq)  # T x n x z
        h_n_seq, hidden_n = self.lstm_enc(e_n_seq)  # T x n x z

        t = random.randint(0, T-1)
        context_width = 2
        c_list = list(range(max(t - context_width, 0), min(t + context_width + 1, T)))
        c_list.remove(t)
        nc_list = list(range(T))
        nc_list.remove(t)
        for i in c_list:
            nc_list.remove(i)
        c_t = random.choice(c_list)
        nc_t = random.choice(nc_list)

        e_seq = torch.cat([e_i_seq, e_p_seq, e_n_seq], dim=1)  # T x 3n x z
        e_t = e_seq[t]
        e_c_t = e_seq[c_t]
        e_nc_t = e_seq[nc_t]

        l_sns = self.loss_sns(h_i_seq[-1], h_p_seq[-1], h_n_seq[-1])
        l_contrast = self.contrast_loss(torch.stack([e_1, e_2], dim=1))
        l_sni = self.contrast_loss(torch.stack([e_t, e_c_t], dim=1))

        loss = 0.
        loss += l_sns * 0.7
        loss += l_contrast * 0.15
        loss += l_sni * 0.15

        metrics = {
            'loss': loss.item(),
            'l_sns': l_sns.item(),
            'l_contrast': l_contrast.item(),
            'l_sin': l_sni.item(),
            'context_sim': F.cosine_similarity(e_t, e_c_t).abs().mean().item(),
            'non_context_sim': F.cosine_similarity(e_t, e_nc_t).abs().mean().item()
        }

        return metrics, loss

    def update(self, video_i, video_p, video_n):

        self.conv_opt.zero_grad()
        self.lstm_enc_opt.zero_grad()

        metrics, loss = self.evaluate(video_i, video_p, video_n)

        loss.backward()

        self.lstm_enc_opt.step()
        self.conv_opt.step()

        return metrics

    def loss_sns(self, h_i, h_p, h_n):
        return F.mse_loss(h_i, h_p) + max(self.rho - F.mse_loss(h_i, h_n), 0.)

    @staticmethod
    def load(file):
        snapshot = Path(file)
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        return payload['encoder']


class LSTMEncoder(nn.Module):
    def __init__(self, input_size):
        super(LSTMEncoder, self).__init__()
        self.num_layers = 2
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=self.num_layers)
        self.fc = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, e_seq):
        T = e_seq.shape[0]
        h_seq, hidden = self.encoder(e_seq)
        h_seq = torch.stack([self.sigmoid(self.fc(h_seq[i])) for i in range(T)])
        return h_seq, hidden
