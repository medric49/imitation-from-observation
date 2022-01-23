from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class CTNet(nn.Module):
    def __init__(self, hidden_dim, lr, lambda_1, lambda_2, lambda_3, use_tb, context_size):
        super(CTNet, self).__init__()
        
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        self.context_size = context_size

        self.use_tb = use_tb

        self.enc1 = EncoderNet(hidden_dim)
        self.enc2 = EncoderNet(hidden_dim)
        self.t = TranslatorNet(hidden_dim)
        self.dec = DecoderNet(hidden_dim)

        self._enc1_opt = torch.optim.Adam(self.enc1.parameters(), lr=lr)
        self._enc2_opt = torch.optim.Adam(self.enc2.parameters(), lr=lr)
        self._t_opt = torch.optim.Adam(self.t.parameters(), lr=lr)
        self._dec_opt = torch.optim.Adam(self.dec.parameters(), lr=lr)
    
    def translate(self, video1, fobs2, return_state=False, keep_enc2=True):
        T = video1.shape[0]

        video1 = video1.to(dtype=torch.float) / 255. - 0.5  # T x c x h x w
        fobs2 = fobs2.to(dtype=torch.float).repeat(T, 1, 1, 1) / 255. - 0.5  # T x c x h x w

        z1, _, _, _, _ = self.enc1(video1)
        fz2, c1, c2, c3, c4 = self.enc2(fobs2)
        z3 = self.t(z1, fz2)

        if keep_enc2:
            z3[0] = fz2[0]

        if return_state:
            return z3

        video2 = self.dec(z3, c1, c2, c3, c4)  # T x c x h x w

        if keep_enc2:
            video2[0] = fobs2[0]
        video2 = (video2 + 0.5) * 255.
        return video2

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

        z_seq = []
        for t in range(T):
            obs1 = video1[t]  # n x c x h x w
            obs2 = video2[t]  # n x c x h x w

            z1, _, _, _, _ = self.enc1(obs1)
            z3 = self.t(z1, fz2)
            z_seq.append(z1)
            z2, _, _, _, _ = self.enc1(obs2)

            l_trans += F.mse_loss(torch.flatten(self.dec(z3, c1, c2, c3, c4), start_dim=1),
                                  torch.flatten(obs2, start_dim=1))
            l_rec += F.mse_loss(torch.flatten(self.dec(z2, c1, c2, c3, c4), start_dim=1),
                                torch.flatten(obs2, start_dim=1))
            l_align += F.mse_loss(z3, z2)

        z_seq = torch.stack(z_seq)  # T x n x z
        z_seq = torch.transpose(z_seq, 0, 1)  # n x T x z

        t1 = torch.randint(0, T - self.context_size, size=(n,))
        t2 = t1 + torch.randint(1, self.context_size, size=(n,))
        t3 = (t1 + torch.randint(self.context_size, T, size=(n, ))) % T

        z_t1 = z_seq[torch.arange(n), t1, :]
        z_t2 = z_seq[torch.arange(n), t2, :]
        z_t3 = z_seq[torch.arange(n), t3, :]

        # l_sim = - torch.log(torch.sigmoid((z_t1 * z_t2).sum(dim=1)) + torch.sigmoid(-(z_t1 * z_t3).sum(dim=1))).mean()
        l_sim = F.cosine_similarity(z_t1, z_t3).abs().sum()

        loss = l_trans + l_rec * self.lambda_1 + l_align * self.lambda_2 + l_sim * self.lambda_3

        return loss, l_trans, l_rec, l_align, l_sim

    def update(self, video1, video2):
        metrics = dict()

        self._enc1_opt.zero_grad()
        self._enc2_opt.zero_grad()
        self._t_opt.zero_grad()
        self._dec_opt.zero_grad()

        loss, l_trans, l_rec, l_align, l_sim = self.evaluate(video1, video2)
        
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
            metrics['sim_loss'] = l_sim.item()

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
        self.translator = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, z1, z2):
        z = torch.cat([z1, z2], dim=1)
        z = self.translator(z)
        return z


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

