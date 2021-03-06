import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from losses import SupConLoss


class ViRLNet(nn.Module):

    def __init__(self, hidden_dim, rho, lr, lambda_1, lambda_2, lambda_3, lambda_4):
        super().__init__()

        self.rho = rho
        self.hidden_dim = hidden_dim

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4

        self.conv = ConvNet(hidden_dim)
        self.deconv = DeconvNet(hidden_dim)
        self.lstm_enc = LSTMEncoder(hidden_dim)
        self.lstm_dec = LSTMDecoder(hidden_dim)

        self.conv_opt = torch.optim.Adam(self.conv.parameters(), lr)
        self.deconv_opt = torch.optim.Adam(self.deconv.parameters(), lr)
        self.lstm_enc_opt = torch.optim.Adam(self.lstm_enc.parameters(), lr)
        self.lstm_dec_opt = torch.optim.Adam(self.lstm_dec.parameters(), lr)

        self.contrast_loss = SupConLoss()

    def encode(self, video):
        video = video.unsqueeze(0)  # 1 x T x c x h x w
        video = video.to(dtype=torch.float) / 255. - 0.5  # 1 x T x c x h x w
        video = torch.transpose(video, dim0=0, dim1=1)  # T x 1 x c x h x w
        e_seq, _, _, _, _ = self._encode(video)
        h, _ = self.lstm_enc(e_seq)  # 1 x h
        h = h[-1][0]
        return h

    def encode_state_seq(self, e_seq):
        e_seq = e_seq.unsqueeze(0)  # 1 x T x z
        e_seq = torch.transpose(e_seq, dim0=0, dim1=1)  # T x 1 x z
        h, _ = self.lstm_enc(e_seq)  # 1 x h
        h = h[-1][0]
        return h

    def encode_frame(self, image):
        shape = image.shape
        image = image.to(dtype=torch.float) / 255. - 0.5
        if len(shape) == 3:
            image = image.unsqueeze(0)  # 1 x c x h x w
        e = self.conv(image)[0]
        if len(shape) == 3:
            return e[0]
        return e

    def encode_decode(self, video):
        T = video.shape[0]
        video = video.unsqueeze(0)  # 1 x T x c x h x w
        video = video.to(dtype=torch.float) / 255. - 0.5  # 1 x T x c x h x w
        video = torch.transpose(video, dim0=0, dim1=1)  # T x 1 x c x h x w

        e_seq, c1_seq, c2_seq, c3_seq, c4_seq = self._encode(video)

        h, hidden = self.lstm_enc(e_seq)

        e0_seq = self.lstm_dec(h, hidden, T)

        video0 = self._decode(e0_seq, c1_seq, c2_seq, c3_seq, c4_seq).squeeze(1)
        video1 = self._decode(e_seq, c1_seq, c2_seq, c3_seq, c4_seq).squeeze(1)

        video0 = self._unnormalize(video0)
        video1 = self._unnormalize(video1)

        return video0, video1

    def _unnormalize(self, video):
        video = (video + 0.5) * 255.
        video[video > 255.] = 255.
        video[video < 0.] = 0.
        return video

    def _encode(self, video):
        e_seq = []
        c1_seq = []
        c2_seq = []
        c3_seq = []
        c4_seq = []
        for t in range(video.shape[0]):
            e, c1, c2, c3, c4 = self.conv(video[t])
            e_seq.append(e)
            c1_seq.append(c1)
            c2_seq.append(c2)
            c3_seq.append(c3)
            c4_seq.append(c4)

        e_seq = torch.stack(e_seq)
        c1_seq = torch.stack(c1_seq)
        c2_seq = torch.stack(c2_seq)
        c3_seq = torch.stack(c3_seq)
        c4_seq = torch.stack(c4_seq)
        return e_seq, c1_seq, c2_seq, c3_seq, c4_seq

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

        video_i = video_i.to(dtype=torch.float) / 255. - 0.5  # n x T x c x h x w
        video_p = video_p.to(dtype=torch.float) / 255. - 0.5  # n x T x c x h x w
        video_n = video_n.to(dtype=torch.float) / 255. - 0.5  # n x T x c x h x w

        video_i = torch.transpose(video_i, dim0=0, dim1=1)  # T x n x c x h x w
        video_p = torch.transpose(video_p, dim0=0, dim1=1)  # T x n x c x h x w
        video_n = torch.transpose(video_n, dim0=0, dim1=1)  # T x n x c x h x w

        e_i_seq, c1_seq_i, c2_seq_i, c3_seq_i, c4_seq_i = self._encode(video_i)
        e_p_seq, c1_seq_p, c2_seq_p, c3_seq_p, c4_seq_p = self._encode(video_p)
        e_n_seq, c1_seq_n, c2_seq_n, c3_seq_n, c4_seq_n = self._encode(video_n)

        h_i_seq, hidden_i = self.lstm_enc(e_i_seq)
        h_p_seq, hidden_p = self.lstm_enc(e_p_seq)
        h_n_seq, hidden_n = self.lstm_enc(e_n_seq)

        e0_i_seq = self.lstm_dec(h_i_seq, hidden_i, T)
        e0_p_seq = self.lstm_dec(h_p_seq, hidden_p, T)

        video0_i = self._decode(e0_i_seq, c1_seq_i, c2_seq_i, c3_seq_i, c4_seq_i)
        video0_p = self._decode(e0_p_seq, c1_seq_p, c2_seq_p, c3_seq_p, c4_seq_p)

        video1_i = self._decode(e_i_seq, c1_seq_i, c2_seq_i, c3_seq_i, c4_seq_i)
        video1_p = self._decode(e_p_seq, c1_seq_p, c2_seq_p, c3_seq_p, c4_seq_p)

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
        # h_seq = torch.cat([h_i_seq, h_p_seq, h_n_seq], dim=1)  # T x 3n x z
        # h_t = h_seq[t]
        # h_c_t = h_seq[c_t]

        l_sns = self.loss_sns(h_i_seq[-1], h_p_seq[-1], h_n_seq[-1])
        # l_sni = self.loss_sni(e_i_seq, e_p_seq, e_n_seq)
        l_sni = self.contrast_loss(torch.stack([e_t, e_c_t], dim=1))
        # l_sni = self.loss_sni(e_t, e_c_t, e_nc_t)
        # l_sni = -(torch.log(torch.sigmoid(torch.inner(e_t, e_c_t))) + torch.log(torch.sigmoid(torch.inner(e_t, -e_nc_t)))).mean()
        # l_sim_seq = self.contrast_loss(torch.stack([h_t, h_c_t], dim=1))
        l_raes = self.loss_vae(video_i, video0_i) + self.loss_vae(video_p, video0_p)
        l_vaei = self.loss_vae(video_i, video1_i) + self.loss_vae(video_p, video1_p)
        loss = 0.7 * l_sns
        loss += 0.1 * l_sni
        # loss += 0.1 * l_sim_seq
        loss += 0.1 * l_raes
        loss += 0.1 * l_vaei

        metrics = {
            'loss': loss.item(),
            'l_sns': l_sns.item(),
            'l_sni': l_sni.item(),
            # 'l_sim_seq': l_sim_seq.item(),
            'l_raes': l_raes.item(),
            'l_vaei': l_vaei.item()
        }

        return metrics, loss

    def update(self, video_i, video_p, video_n):

        self.conv_opt.zero_grad()
        self.lstm_enc_opt.zero_grad()
        self.lstm_dec_opt.zero_grad()
        self.deconv_opt.zero_grad()

        metrics, loss = self.evaluate(video_i, video_p, video_n)

        loss.backward()

        self.deconv_opt.step()
        self.lstm_dec_opt.step()
        self.lstm_enc_opt.step()
        self.conv_opt.step()

        return metrics

    def loss_sns(self, h_i, h_p, h_n):
        return F.mse_loss(h_i, h_p) + max(self.rho - F.mse_loss(h_i, h_n), 0.)

    def loss_sni(self, e_seq_i, e_seq_p, e_seq_n):
        T = e_seq_i.shape[0]
        l = 0.
        for t in range(T):
            l += F.mse_loss(e_seq_i[t], e_seq_p[t]) + max(self.rho - F.mse_loss(e_seq_i[t], e_seq_n[t]), 0.)
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


class ConvNet(nn.Module):
    def __init__(self, hidden_dim):
        super(ConvNet, self).__init__()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=5, stride=2)
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


class DeconvNet(nn.Module):
    def __init__(self, hidden_dim):
        super(DeconvNet, self).__init__()
        self.leaky_relu = nn.LeakyReLU()

        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1)
        self.fc1 = nn.Conv2d(hidden_dim * 4, 512, kernel_size=1)
        self.b_norm_fc_1 = nn.BatchNorm2d(512)
        # self.conn_4 = nn.Conv2d(512 * 2, 512, kernel_size=1, stride=1)

        self.t_conv_4 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)
        self.b_norm_4 = nn.BatchNorm2d(256)
        # self.conn_3 = nn.Conv2d(256 * 2, 256, kernel_size=1, stride=1)

        self.t_conv_3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.b_norm_3 = nn.BatchNorm2d(128)
        # self.conn_2 = nn.Conv2d(128 * 2, 128, kernel_size=1, stride=1)

        self.t_conv_2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, output_padding=1)
        self.b_norm_2 = nn.BatchNorm2d(64)
        # self.conn_1 = nn.Conv2d(64 * 2, 64, kernel_size=1, stride=1)

        self.t_conv_1 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, output_padding=1)

    def forward(self, e, c1, c2, c3, c4):
        e = e.view(e.shape[0], e.shape[1], 1, 1)
        e = self.leaky_relu(self.fc2(e))

        d4 = self.leaky_relu(self.b_norm_fc_1(self.fc1(e)))
        # d4 = self.leaky_relu(self.conn_4(torch.cat([c4, d4], dim=1)))

        d3 = self.leaky_relu(self.b_norm_4(self.t_conv_4(d4)))
        # d3 = self.leaky_relu(self.conn_3(torch.cat([c3, d3], dim=1)))

        d2 = self.leaky_relu(self.b_norm_3(self.t_conv_3(d3)))
        # d2 = self.leaky_relu(self.conn_2(torch.cat([c2, d2], dim=1)))

        d1 = self.leaky_relu(self.b_norm_2(self.t_conv_2(d2)))
        # d1 = self.leaky_relu(self.conn_1(torch.cat([c1, d1], dim=1)))

        obs = self.leaky_relu(self.t_conv_1(d1))
        return obs


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

    def forward(self, h_seq, hidden, T):
        # hidden = None
        # h = h_seq[-1]
        # e_seq = []
        # for _ in range(T):
        #     h = h.unsqueeze(0)
        #     h, hidden = self.decoder(h, hidden)
        #     h = h.squeeze(0)
        #     e_seq.append(h)
        # e_seq = torch.stack(e_seq)
        # return e_seq
        return self.decoder(h_seq)[0]
