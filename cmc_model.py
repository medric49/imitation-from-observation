import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from hydra.utils import to_absolute_path

import alexnet
from losses import SupConLoss


class OneSideContrastLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(OneSideContrastLoss, self).__init__()
        self.temperature = temperature

    def forward(self, h_i, h_p, h_n_sample):
        nb_negative = h_n_sample.shape[0]
        h_i = h_i.unsqueeze(0)
        h_p = h_p.unsqueeze(0)

        sim_1 = torch.exp(F.cosine_similarity(h_i, h_p) * (1. / self.temperature)).sum()
        sim_2 = torch.exp(F.cosine_similarity(h_i.repeat([nb_negative, 1]), h_n_sample) * (1. / self.temperature)).sum()

        loss = -torch.log(sim_1 / (sim_1 + sim_2))

        return loss


class ConvNet224(nn.Module):
    def __init__(self, hidden_dim):
        super(ConvNet224, self).__init__()

        self.alex_net = alexnet.MyAlexNetCMC()
        self.alex_net.load_state_dict(torch.load(to_absolute_path('tmp/CMC_alexnet.pth'))['model'])

        self.fc_l = nn.Sequential(
            nn.Linear(128, hidden_dim // 2),
            nn.Sigmoid()
        )

        self.fc_ab = nn.Sequential(
            nn.Linear(128, hidden_dim // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_l, x_ab = self.alex_net(x)
        x_l = self.fc_l(x_l)
        x_ab = self.fc_ab(x_ab)
        return x_l, x_ab


class HalfConvNet(nn.Module):
    def __init__(self, in_channel, hidden_dim):
        super(HalfConvNet, self).__init__()
        self.leaky_relu = nn.LeakyReLU(inplace=True)
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
        e = self.leaky_relu(self.b_norm_1(self.conv_1(obs)))
        e = self.leaky_relu(self.b_norm_2(self.conv_2(e)))
        e = self.leaky_relu(self.b_norm_3(self.conv_3(e)))
        e = self.leaky_relu(self.b_norm_4(self.conv_4(e)))
        e = self.leaky_relu(self.b_norm_fc_1(self.fc1(e)))
        e = self.sigmoid(self.fc2(e))
        e = e.view(e.shape[0], e.shape[1])
        return e


class DeconvNet(nn.Module):
    def __init__(self, hidden_dim):
        super(DeconvNet, self).__init__()
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1)
        self.fc1 = nn.Conv2d(hidden_dim * 4, 512, kernel_size=1)
        self.b_norm_fc_1 = nn.BatchNorm2d(512)

        self.t_conv_4 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)
        self.b_norm_4 = nn.BatchNorm2d(256)

        self.t_conv_3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.b_norm_3 = nn.BatchNorm2d(128)

        self.t_conv_2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, output_padding=1)
        self.b_norm_2 = nn.BatchNorm2d(64)

        self.t_conv_1 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, output_padding=1)

    def forward(self, e):
        e = e.view(e.shape[0], e.shape[1], 1, 1)
        e = self.leaky_relu(self.fc2(e))
        d4 = self.leaky_relu(self.b_norm_fc_1(self.fc1(e)))
        d3 = self.leaky_relu(self.b_norm_4(self.t_conv_4(d4)))
        d2 = self.leaky_relu(self.b_norm_3(self.t_conv_3(d3)))
        d1 = self.leaky_relu(self.b_norm_2(self.t_conv_2(d2)))
        obs = self.t_conv_1(d1)
        return obs


class ConvNet(nn.Module):
    def __init__(self, hidden_dim):
        super(ConvNet, self).__init__()

        self.enc_l = HalfConvNet(1, hidden_dim // 2)
        self.enc_ab = HalfConvNet(2, hidden_dim // 2)

    def forward(self, x):
        x_l, x_ab = torch.split(x, [1, 2], dim=1)
        x_l = self.enc_l(x_l)
        x_ab = self.enc_ab(x_ab)

        return x_l, x_ab


class CMCModel(nn.Module):

    def __init__(self, hidden_dim, rho, lr):
        super(CMCModel, self).__init__()

        self.rho = rho
        self.hidden_dim = hidden_dim
        self.conv = ConvNet(hidden_dim)
        self.deconv = DeconvNet(hidden_dim)
        self.lstm_enc = LSTMEncoder(hidden_dim)
        self.lstm_dec = LSTMDecoder(hidden_dim)

        self.conv_opt = torch.optim.Adam(self.conv.parameters(), lr)
        self.deconv_opt = torch.optim.Adam(self.deconv.parameters(), lr)
        self.lstm_enc_opt = torch.optim.Adam(self.lstm_enc.parameters(), lr)
        self.lstm_dec_opt = torch.optim.Adam(self.lstm_dec.parameters(), lr)

        self.contrast_loss = SupConLoss()
        self.one_side_contrast_loss = OneSideContrastLoss()

    def encode(self, video):
        e_seq = self.encode_frame(video)
        h, _ = self.lstm_enc(e_seq)
        return h[-1]

    def encode_state_seq(self, e_seq):
        e_seq = e_seq.unsqueeze(1)  # T x 1 x z
        h_seq, _ = self.lstm_enc(e_seq)
        return h_seq.squeeze(1)

    def encode_frame(self, image):
        shape = image.shape
        if len(shape) == 3:
            image = image.unsqueeze(0)  # 1 x c x h x w
        e_1, e_2 = self.conv(image)
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
            e_1, e_2 = self.conv(frame)
            e_1_seq.append(e_1)
            e_2_seq.append(e_2)
        e_1_seq = torch.stack(e_1_seq)  # T x n x z/2
        e_2_seq = torch.stack(e_2_seq)  # T x n x z/2
        return e_1_seq, e_2_seq

    def _decode(self, e_seq):
        video = []
        for t in range(e_seq.shape[0]):
            o = self.deconv(e_seq[t])
            video.append(o)
        video = torch.stack(video)
        return video

    def evaluate(self, video):
        T = video.shape[1]
        n = video.shape[0]

        video = torch.transpose(video, dim0=0, dim1=1)  # T x n x c x h x w

        e_1_seq, e_2_seq = self._encode(video)  # T x n x z/2

        e_seq = torch.cat([e_1_seq, e_2_seq], dim=2)  # T x n x z

        h_seq, hidden = self.lstm_enc(e_seq)  # T x n x z
        e0_seq = self.lstm_dec(h_seq)
        video0 = self._decode(e_seq)

        t = random.randint(0, T-1)
        context_width = 1
        c_list = list(range(max(t - context_width, 0), min(t + context_width + 1, T)))
        c_list.remove(t)
        nc_list = list(range(T))
        nc_list.remove(t)
        for i in c_list:
            nc_list.remove(i)
        c_t = random.choice(c_list)
        nc_t = random.choice(nc_list)

        e_t = e_seq[t]
        e_c_t = e_seq[c_t]
        e_nc_t = e_seq[nc_t]

        h_t = h_seq[t]
        h_c_t = h_seq[c_t]
        h_nc_t = h_seq[nc_t]

        h_i = h_seq[:, 0, :][t]  # z
        h_p = h_seq[:, 1, :][t]  # z
        h_n_samples = h_seq[:, 2:, :][t]  # (n-2) x z

        l_sns = self.one_side_contrast_loss(h_i, h_p, h_n_samples) + self.one_side_contrast_loss(h_p, h_i, h_n_samples)
        l_frame = self.contrast_loss(torch.stack([e_1_seq.view(n * T, -1), e_2_seq.view(n * T, -1)], dim=1)) + self.loss_sns(e_t, e_c_t, e_nc_t)  # + self.contrast_loss(torch.stack([e_t, e_c_t], dim=1))
        l_seq = self.loss_sns(h_t, h_c_t, h_nc_t)
        l_vaes = self.loss_vae_seq(e_seq, e0_seq)
        l_vaei = self.loss_vae(video, video0)

        loss = 0.
        loss += l_sns * 0.6
        loss += l_frame * 0.1
        loss += l_seq * 0.1
        loss += l_vaes * 0.1
        loss += l_vaei * 0.1

        metrics = {
            'loss': loss.item(),
            'l_sns': l_sns.item(),
            'l_frame': l_frame.item(),
            'l_seq': l_seq.item(),
            'l_vaes': l_vaes.item(),
            'l_vaei': l_vaei.item(),
            'context_sim': F.cosine_similarity(h_t, h_c_t).abs().mean().item(),
            'non_context_sim': F.cosine_similarity(h_t, h_nc_t).abs().mean().item()
        }

        return metrics, loss

    def update(self, video):

        self.conv_opt.zero_grad()
        self.lstm_enc_opt.zero_grad()
        self.lstm_dec_opt.zero_grad()
        self.deconv_opt.zero_grad()

        metrics, loss = self.evaluate(video)
        loss = torch.nan_to_num(loss, -1)
        if loss.item() == -1:
            print('***')
            return metrics

        loss.backward()

        self.deconv_opt.step()
        self.lstm_enc_opt.step()
        self.lstm_dec_opt.step()
        self.conv_opt.step()

        return metrics

    def loss_sns(self, h_i, h_p, h_n):
        return F.mse_loss(h_i, h_p) + max(self.rho - F.mse_loss(h_i, h_n), 0.)

    def loss_vae_seq(self, e_seq, e0_seq):
        T = e_seq.shape[0]
        l = 0.
        for t in range(T):
            l += F.mse_loss(e_seq[t], e0_seq[t])
        l /= T
        return l

    def loss_vae(self, video1, video2):
        T = video1.shape[0]
        l = 0.
        for t in range(T):
            o1 = video1[t].flatten(start_dim=1)
            o2 = video2[t].flatten(start_dim=1)
            l += F.mse_loss(o1, o2)
        l /= T
        return l

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


class LSTMDecoder(nn.Module):
    def __init__(self, input_size):
        super(LSTMDecoder, self).__init__()
        self.num_layers = 2
        self.decoder = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=self.num_layers)
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, h_seq):
        T = h_seq.shape[0]
        h_seq = torch.stack([self.fc(h_seq[i]) for i in range(T)])
        return self.decoder(h_seq)[0]
