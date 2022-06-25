from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from losses import SupConLoss


class HalfAlexNet(nn.Module):
    def __init__(self, in_channel, state_dim):
        super(HalfAlexNet, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 48, 3, 1, 1, bias=False),  # 64 -> 64
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 64 -> 31
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(48, 96, 3, 1, 1, bias=False),  # 31 -> 31
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 31 -> 15
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(96, 192, 3, 1, 1, bias=False),  # 15 -> 15
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),  # 15 -> 15
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(192, 96, 3, 1, 1, bias=False),  # 15 -> 15
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 15 ->  7
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 7 * 7, state_dim),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, hidden_dim):
        super(ConvNet, self).__init__()

        self.enc_l = HalfAlexNet(1, hidden_dim)
        self.enc_ab = HalfAlexNet(2, hidden_dim)

    def forward(self, x_l, x_ab):
        x_l = self.enc_l(x_l)
        x_ab = self.enc_ab(x_ab)

        return x_l, x_ab


class CMCModel(nn.Module):

    def __init__(self, hidden_dim, rho, lr):
        super(CMCModel, self).__init__()

        self.rho = rho

        self.conv = ConvNet(hidden_dim)
        self.lstm_enc = LSTMEncoder(hidden_dim)

        self.conv_opt = torch.optim.Adam(self.conv.parameters(), lr)
        self.lstm_enc_opt = torch.optim.Adam(self.lstm_enc.parameters(), lr)

        self._contrast_loss = SupConLoss(
            contrast_mode='all',
            temperature=0.05,
            base_temperature=0.05
        )

    def encode(self, video):
        e_seq = self.encode_frame(video)
        h, _ = self.lstm_enc(e_seq)
        return h

    def encode_state_seq(self, e_seq):
        h, _ = self.lstm_enc(e_seq)
        return h

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
        view_1_seq, view_2_seq = torch.split(video, [1, 2], dim=2)
        view_1_seq = view_1_seq.view(-1, 1, shape[3], shape[4])  # Tn x c x h x w
        view_2_seq = view_2_seq.view(-1, 2, shape[3], shape[4])  # Tn x c x h x w
        return self.conv(view_1_seq, view_2_seq)

    def _decode(self, e_seq, c1_seq, c2_seq, c3_seq, c4_seq):
        video = []
        for t in range(e_seq.shape[0]):
            o = self.deconv(e_seq[t], c1_seq[t], c2_seq[t], c3_seq[t], c4_seq[t])
            video.append(o)
        video = torch.stack(video)
        return video

    def evaluate(self, video_i, video_p, video_n):
        T = video_i.shape[1]
        n = video_i.shape[0]

        video_i = torch.transpose(video_i, dim0=0, dim1=1)  # T x n x c x h x w
        video_p = torch.transpose(video_p, dim0=0, dim1=1)  # T x n x c x h x w
        video_n = torch.transpose(video_n, dim0=0, dim1=1)  # T x n x c x h x w

        e_i1, e_i2 = self._encode(video_i)  # Tn x z
        e_p1, e_p2 = self._encode(video_p)  # Tn x z
        e_n1, e_n2 = self._encode(video_n)  # Tn x z

        e_1 = torch.cat([e_i1, e_p1, e_n1], dim=0)  # 3Tn x z
        e_2 = torch.cat([e_i2, e_p2, e_n2], dim=0)  # 3Tn x z

        e_i_seq = torch.cat([e_i1, e_i2], dim=1).view(T, n, -1)  # T x n x 2z
        e_p_seq = torch.cat([e_p1, e_p2], dim=1).view(T, n, -1)  # T x n x 2z
        e_n_seq = torch.cat([e_n1, e_n2], dim=1).view(T, n, -1)  # T x n x 2z

        h_i, hidden_i = self.lstm_enc(e_i_seq)
        h_p, hidden_p = self.lstm_enc(e_p_seq)
        h_n, hidden_n = self.lstm_enc(e_n_seq)

        l_sns = self.loss_sns(h_i, h_p, h_n)
        l_contrast = self._contrast_loss(torch.stack([e_1, e_2], dim=1))

        loss = 0.
        loss += l_sns * 0.5
        loss += l_contrast * 0.5

        metrics = {
            'loss': loss.item(),
            'l_sns': l_sns.item(),
            'l_contrast': l_contrast.item(),
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
    def __init__(self, state_dim):
        super(LSTMEncoder, self).__init__()
        self.num_layers = 2
        self.encoder = nn.LSTM(input_size=state_dim * 2, hidden_size=state_dim, num_layers=self.num_layers)
        self.fc = nn.Linear(state_dim, state_dim)

    def forward(self, e_seq):
        h_seq, hidden = self.encoder(e_seq)
        h = h_seq[-1]
        h = self.fc(h)
        return h, hidden

