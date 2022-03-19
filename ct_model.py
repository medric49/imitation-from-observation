from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class CTNet(nn.Module):
    def __init__(self, hidden_dim, lr, lambda_trans, lambda_rec, lambda_align, use_tb, translator):
        super(CTNet, self).__init__()

        self.lambda_trans = lambda_trans
        self.lambda_rec = lambda_rec
        self.lambda_align = lambda_align

        self.use_tb = use_tb

        self.enc1 = EncoderNet(hidden_dim)
        self.enc2 = EncoderNet(hidden_dim)
        self.t = translator
        self.dec = DecoderNet(hidden_dim)

        self._enc1_opt = torch.optim.Adam(self.enc1.parameters(), lr=lr)
        self._enc2_opt = torch.optim.Adam(self.enc2.parameters(), lr=lr)
        self._t_opt = torch.optim.Adam(self.t.parameters(), lr=lr)
        self._dec_opt = torch.optim.Adam(self.dec.parameters(), lr=lr)
    
    def translate(self, video1, fobs2, keep_enc2=True):
        T = video1.shape[0]

        video1 = video1.to(dtype=torch.float) / 255. - 0.5  # T x c x h x w
        fobs2 = fobs2.to(dtype=torch.float) / 255. - 0.5  # c x h x w
        video1 = video1.unsqueeze(dim=1)
        fobs2 = fobs2.unsqueeze(dim=0)

        z1_seq = [self.enc1(video1[t])[0] for t in range(T)]
        z1_seq = torch.stack(z1_seq)

        fz2, c1, c2, c3, c4 = self.enc2(fobs2)

        z3_seq = self.t(z1_seq, fz2)

        video2 = [self.dec(z3_seq[t], c1, c2, c3, c4) for t in range(T)]  # T x c x h x w
        video2 = torch.stack(video2)

        z3_seq = z3_seq.squeeze(dim=1)
        video2 = video2.squeeze(dim=1)
        if keep_enc2:
            video2[0] = fobs2[0]
        video2 = (video2 + 0.5) * 255.
        video2[video2 > 255.] = 255.
        video2[video2 < 0.] = 0.
        return z3_seq, video2

    def evaluate(self, video1, video2):
        T = video1.shape[1]
        n = video1.shape[0]

        video1 = video1.to(dtype=torch.float) / 255. - 0.5  # n x T x c x h x w
        video2 = video2.to(dtype=torch.float) / 255. - 0.5  # n x T x c x h x w

        video1 = torch.transpose(video1, dim0=0, dim1=1)  # T x n x c x h x w
        video2 = torch.transpose(video2, dim0=0, dim1=1)  # T x n x c x h x w

        fobs2 = video2[0]  # n x c x h x w

        l_trans = 0
        l_rec = 0
        l_align = 0

        fz2, c1, c2, c3, c4 = self.enc2(fobs2)

        z1_seq = [self.enc1(video1[t])[0] for t in range(T)]
        z1_seq = torch.stack(z1_seq)
        z3_seq = self.t(z1_seq, fz2)

        z2_seq = [self.enc1(video2[t])[0] for t in range(T)]
        z2_seq = torch.stack(z2_seq)

        for t in range(T):
            obs2 = video2[t]
            obs_z3 = self.dec(z3_seq[t], c1, c2, c3, c4)
            obs_z2 = self.dec(z2_seq[t], c1, c2, c3, c4)

            l_trans += F.mse_loss(torch.flatten(obs_z3, start_dim=1), torch.flatten(obs2, start_dim=1))
            l_rec += F.mse_loss(torch.flatten(obs_z2, start_dim=1), torch.flatten(obs2, start_dim=1))
            l_align += F.mse_loss(z3_seq[t], z2_seq[t])

        loss = l_trans * self.lambda_trans + l_rec * self.lambda_rec + l_align * self.lambda_align

        return loss, l_trans, l_rec, l_align

    def update(self, video1, video2):
        metrics = dict()

        self._enc1_opt.zero_grad()
        self._enc2_opt.zero_grad()
        self._t_opt.zero_grad()
        self._dec_opt.zero_grad()

        loss, l_trans, l_rec, l_align = self.evaluate(video1, video2)
        
        loss.backward()
        
        self._dec_opt.step()
        self._t_opt.step()
        self._enc2_opt.step()
        self._enc1_opt.step()

        if self.use_tb:
            metrics['loss'] = loss.item()
            metrics['trans_loss'] = l_trans.item()
            metrics['rec_loss'] = l_rec.item()
            metrics['align_loss'] = l_align.item()

        return metrics

    def encode(self, obs):
        obs = obs.to(dtype=torch.float) / 255. - 0.5
        obs, _, _, _, _ = self.enc1(obs)
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
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.flatten = nn.Flatten()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=5, stride=2)
        self.b_norm_1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.b_norm_2 = nn.BatchNorm2d(128)
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.b_norm_3 = nn.BatchNorm2d(256)
        self.conv_4 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.b_norm_4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.b_norm_fc_1 = nn.BatchNorm2d(hidden_dim)
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, obs):
        c1 = self.leaky_relu(self.b_norm_1(self.conv_1(obs)))
        c2 = self.leaky_relu(self.b_norm_2(self.conv_2(c1)))
        c3 = self.leaky_relu(self.b_norm_3(self.conv_3(c2)))
        c4 = self.leaky_relu(self.b_norm_4(self.conv_4(c3)))
        z = self.leaky_relu(self.b_norm_fc_1(self.fc1(c4)))
        z = self.leaky_relu(self.fc2(z))
        z = z.view(z.shape[0], z.shape[1])
        return z, c1, c2, c3, c4


class TranslatorNet(nn.Module):
    def __init__(self, hidden_dim):
        super(TranslatorNet, self).__init__()
        self.translator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, z1_seq, fz2):
        z3_seq = []
        for z1 in z1_seq:
            z3 = self.translator(torch.cat([z1, fz2], dim=1))
            z3_seq.append(z3)
        z3_seq = torch.stack(z3_seq)
        return z3_seq


class DecoderNet(nn.Module):
    def __init__(self, hidden_dim):
        super(DecoderNet, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.fc = nn.Conv2d(hidden_dim, 512, kernel_size=1)
        self.b_norm_fc = nn.BatchNorm2d(512)
        self.conn_4 = nn.Conv2d(512 * 2, 512, kernel_size=1, stride=1)

        self.t_conv_4 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)
        self.b_norm_4 = nn.BatchNorm2d(256)
        self.conn_3 = nn.Conv2d(256 * 2, 256, kernel_size=1, stride=1)

        self.t_conv_3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.b_norm_3 = nn.BatchNorm2d(128)
        self.conn_2 = nn.Conv2d(128 * 2, 128, kernel_size=1, stride=1)

        self.t_conv_2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, output_padding=1)
        self.b_norm_2 = nn.BatchNorm2d(64)
        self.conn_1 = nn.Conv2d(64 * 2, 64, kernel_size=1, stride=1)

        self.t_conv_1 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, output_padding=1)

    def forward(self, z, c1, c2, c3, c4):
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        d4 = self.leaky_relu(self.b_norm_fc(self.fc(z)))
        d4 = self.leaky_relu(self.conn_4(torch.cat([c4, d4], dim=1)))

        d3 = self.leaky_relu(self.b_norm_4(self.t_conv_4(d4)))
        d3 = self.leaky_relu(self.conn_3(torch.cat([c3, d3], dim=1)))

        d2 = self.leaky_relu(self.b_norm_3(self.t_conv_3(d3)))
        d2 = self.leaky_relu(self.conn_2(torch.cat([c2, d2], dim=1)))

        d1 = self.leaky_relu(self.b_norm_2(self.t_conv_2(d2)))
        d1 = self.leaky_relu(self.conn_1(torch.cat([c1, d1], dim=1)))

        obs = self.leaky_relu(self.t_conv_1(d1))
        return obs


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
