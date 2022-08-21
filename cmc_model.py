import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import alexnet
import nets
import utils
from losses import SupConLoss

PATCH_SIZE = 50


class LSTMEncoder(nn.Module):
    def __init__(self, input_size):
        super(LSTMEncoder, self).__init__()
        self.num_layers = 2
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=self.num_layers)
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, e_seq):
        T = e_seq.shape[0]
        h_seq, hidden = self.encoder(e_seq)
        h_seq = torch.stack([self.fc(h_seq[i]) for i in range(T)])
        return h_seq, hidden


class LSTMDecoder(nn.Module):
    def __init__(self, input_size):
        super(LSTMDecoder, self).__init__()
        self.num_layers = 2
        self.decoder = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=self.num_layers)
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, h, T):
        hidden = None
        e_seq = []
        h = h.unsqueeze(0)
        for t in range(T):
            h, hidden = self.decoder(h, hidden)
            e_seq.append(self.fc(h[0]))
        e_seq = torch.stack(e_seq)
        return e_seq


class Predictor(nn.Module):
    def __init__(self, input_size):
        super(Predictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, input_size),
        )

    def forward(self, h):
        return self.net(h)


class OneSideContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(OneSideContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, h_i, h_p, h_n_samples):
        if len(h_i.shape) == 1:
            h_i = h_i.unsqueeze(0)  # 1 x z
            h_p = h_p.unsqueeze(0)  # 1 x z
            h_n_samples = h_n_samples.unsqueeze(1)  # nb x 1 x z
        nb_negative = h_n_samples.shape[0]  # nb
        h_n_samples = h_n_samples.transpose(1, 0)  # n x nb x z
        h_i_repeat = h_i.repeat([nb_negative, 1, 1]).transpose(1, 0)  # n x nb x z

        sim_1 = torch.exp(F.cosine_similarity(h_i, h_p) * (1. / self.temperature))  # n
        sim_2 = torch.exp(F.cosine_similarity(h_i_repeat, h_n_samples, dim=2) * (1. / self.temperature))  # n x nb
        sim_2 = sim_2.sum(dim=1)  # n
        loss = -torch.log(sim_1 / (sim_1 + sim_2))
        loss = loss.mean()
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.loss = SupConLoss()
        self.normalize = alexnet.Normalize()

    def forward(self, x1, x2):
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)
        x = torch.stack([x1, x2], dim=1)
        return self.loss(x)


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
        e = self.fc2(e)
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

    def encode(self, video):
        e_seq = self.encode_frame(video)
        h, _ = self.lstm_enc(e_seq)
        return h[-1]

    def encode_state_seq(self, e_seq):
        e_seq = e_seq.unsqueeze(1)  # T x 1 x z
        h_seq, _ = self.lstm_enc(e_seq)
        return h_seq.squeeze(1)

    def encode_frame(self, image):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def deactivate_state_update(self):
        raise NotImplementedError

    def evaluate_seq_encoder(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        seq_only = kwargs.get('seq_only', False)

        self.train()
        self.optimizer.zero_grad()

        if not seq_only:
            metrics, loss = self.evaluate(*args, **kwargs)
        else:
            metrics, loss = self.evaluate_seq_encoder(*args, **kwargs)

        loss = torch.nan_to_num(loss, -1)
        if loss.item() == -1:
            print('***')
            return metrics

        loss.backward()

        self.optimizer.step()
        self.eval()
        return metrics

    def loss_sns(self, h_i, h_p, h_n):
        return F.mse_loss(h_i, h_p) + max(self.rho - F.mse_loss(h_i, h_n), 0.)

    def loss_vae(self, e_seq, e0_seq):
        T = e_seq.shape[0]
        l = 0.
        for t in range(T):
            l += F.mse_loss(e_seq[t], e0_seq[t])
        l /= T
        return l

    def loss_vae_img(self, video1, video2):
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


class CMCBasic(CMCModel):
    def __init__(self, hidden_dim, rho, lr):
        super(CMCBasic, self).__init__()
        self.rho = rho
        self.hidden_dim = hidden_dim
        self.conv = ConvNet(hidden_dim)
        self.deconv = DeconvNet(hidden_dim)
        self.lstm_enc = LSTMEncoder(hidden_dim)
        self.predictor = Predictor(hidden_dim)

        self.optimizer = torch.optim.Adam(
            list(self.conv.parameters()) + list(self.deconv.parameters()) + list(self.lstm_enc.parameters()) + list(
                self.predictor.parameters()), lr)

        self.contrast_loss = ContrastiveLoss()
        self.one_side_contrast_loss = OneSideContrastiveLoss()

    def deactivate_state_update(self):
        for param in self.conv.parameters():
            param.requires_grad = False

    def encode_frame(self, image):
        shape = image.shape
        if len(shape) == 3:
            image = image.unsqueeze(0)  # 1 x c x h x w
        e_1, e_2 = self.conv(image)
        e = torch.cat([e_1, e_2], dim=1)
        if len(shape) == 3:
            e = e.squeeze()
        return e

    def compute_seq_losses(self, e_seq, T, n):
        h_seq, hidden = self.lstm_enc(e_seq)  # T x 2n x z
        h_i_seq = h_seq[:, :n, :]
        h_n_seq = h_seq[:, n:, :]
        l_seq = 0.
        for i in range(n):
            j = (i + 1) % n
            l_seq += self.one_side_contrast_loss(h_i_seq[-1, i], h_i_seq[-1, j],
                                                 h_n_seq[-1]) + self.one_side_contrast_loss(h_n_seq[-1, i],
                                                                                            h_n_seq[-1, j], h_i_seq[-1])
        l_seq /= n

        l_vaes = 0.
        future_len = 2
        k = random.randint(5, T - future_len - 1)
        e_seq_pred = e_seq[:k + 1]
        for i in range(k, k + future_len):
            h = self.lstm_enc(e_seq_pred)[0][-1]
            e_next_pred = self.predictor(h)
            e_next_true = e_seq[i + 1]
            e_neg_sample = torch.cat([e_seq[:i + 1], e_seq[i + 2:]])
            l_vaes += self.one_side_contrast_loss(e_next_pred, e_next_true, e_neg_sample)
            e_seq_pred = torch.cat([e_seq_pred, e_next_pred.unsqueeze(0)])
        l_vaes /= future_len
        return l_seq, l_vaes

    def compute_img_losses(self, video, e_seq, e_1_seq, e_2_seq, T):
        t, c_t, nc_t = utils.context_indices(T, context_width=2)
        e_t = e_seq[t]
        e_c_t = e_seq[c_t]
        e_nc_t = e_seq[nc_t]
        l_frame = self.contrast_loss(e_1_seq.view(-1, self.hidden_dim), e_2_seq.view(-1, self.hidden_dim)) + self.loss_sns(e_t, e_c_t, e_nc_t) + self.contrast_loss(e_t, e_c_t)

        video0 = self._decode(e_seq)
        l_vaei = 0.
        for i in range(0, T, PATCH_SIZE):
            patch = video[i: i + PATCH_SIZE]
            patch = patch.to(device=utils.device(), dtype=torch.float)
            patch0 = video0[i: i + PATCH_SIZE]
            patch0 = patch0.to(device=utils.device(), dtype=torch.float)
            l_vaei += self.loss_vae_img(patch, patch0) * patch0.shape[0]
        l_vaei /= T
        return l_frame, l_vaei

    def evaluate_seq_encoder(self, video_i, video_n, *args, **kwargs):
        T = video_i.shape[1]
        n = video_i.shape[0]

        video_i = torch.transpose(video_i, dim0=0, dim1=1)  # T x n x c x h x w
        video_n = torch.transpose(video_n, dim0=0, dim1=1)  # T x n x c x h x w
        video = torch.cat([video_i, video_n], dim=1)  # T x 2n x c x h x w

        e_1_seq = []
        e_2_seq = []
        for i in range(0, T, PATCH_SIZE):
            patch = video[i: i + PATCH_SIZE]
            patch = patch.to(device=utils.device(), dtype=torch.float)
            e_1, e_2 = self._encode(patch)
            e_1_seq.append(e_1)
            e_2_seq.append(e_2)

        e_1_seq = torch.cat(e_1_seq)
        e_2_seq = torch.cat(e_2_seq)
        e_seq = torch.cat([e_1_seq, e_2_seq], dim=2)  # T x 2n x z

        l_seq, l_vaes = self.compute_seq_losses(e_seq, T, n)

        loss = 0.
        loss += l_seq * 0.5
        loss += l_vaes * 0.5

        metrics = {
            'loss': loss.item(),
            'l_seq': l_seq.item(),
            'l_vaes': l_vaes.item(),
        }

        return metrics, loss

    def evaluate(self, video_i, video_n, *args, **kwargs):
        T = video_i.shape[1]
        n = video_i.shape[0]

        video_i = torch.transpose(video_i, dim0=0, dim1=1)  # T x n x c x h x w
        video_n = torch.transpose(video_n, dim0=0, dim1=1)  # T x n x c x h x w
        video = torch.cat([video_i, video_n], dim=1)  # T x 2n x c x h x w

        e_1_seq = []
        e_2_seq = []
        for i in range(0, T, PATCH_SIZE):
            patch = video[i: i + PATCH_SIZE]
            patch = patch.to(device=utils.device(), dtype=torch.float)
            e_1, e_2 = self._encode(patch)
            e_1_seq.append(e_1)
            e_2_seq.append(e_2)

        e_1_seq = torch.cat(e_1_seq)
        e_2_seq = torch.cat(e_2_seq)
        e_seq = torch.cat([e_1_seq, e_2_seq], dim=2)  # T x 2n x z

        l_seq, l_vaes = self.compute_seq_losses(e_seq, T, n)
        l_frame, l_vaei = self.compute_img_losses(video, e_seq, e_1_seq, e_2_seq, T)

        loss = 0.
        loss += l_seq * 0.25
        loss += l_frame * 0.25
        loss += l_vaes * 0.25
        loss += l_vaei * 0.25

        metrics = {
            'loss': loss.item(),
            'l_seq': l_seq.item(),
            'l_frame': l_frame.item(),
            'l_vaes': l_vaes.item(),
            'l_vaei': l_vaei.item()
        }

        return metrics, loss

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
            o = self.deconv(e_seq[t]).to(device=torch.device('cpu'))
            video.append(o)
        video = torch.stack(video)
        return video


class CMCImgEncoder(CMCModel):
    def __init__(self, hidden_dim, rho, lr):
        super(CMCImgEncoder, self).__init__()

        self.rho = rho
        self.hidden_dim = hidden_dim
        self.img_encoder = nets.EfficientNetB0Encoder()
        self.conv = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LeakyReLU(),
            nn.Linear(512, hidden_dim),
            alexnet.Normalize(),
        )
        self.deconv = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1280)
        )
        self.lstm_enc = LSTMEncoder(hidden_dim)
        self.lstm_dec = LSTMDecoder(hidden_dim)

        self.optimizer = torch.optim.Adam(list(self.conv.parameters()) + list(self.deconv.parameters()) + list(self.lstm_enc.parameters()) + list(self.lstm_dec.parameters()), lr)

    def train(self, *args, **kwargs):
        super(CMCModel, self).train(*args, **kwargs)
        self.img_encoder.eval()

    def _encode_video(self, video):
        T = video.shape[0]  # T x n x c x h x w
        s_seq = []
        for t in range(T):
            frame = video[t]  # n x c x h x w
            s = self.img_encoder(frame)
            s_seq.append(s)
        s_seq = torch.stack(s_seq)  # T x n x z
        return s_seq

    def _encode(self, s_seq):
        T = s_seq.shape[0]  # T x n x s
        e_seq = []
        for t in range(T):
            e_seq.append(self.conv(s_seq[t]))
        e_seq = torch.stack(e_seq)  # T x n x z
        return e_seq

    def _decode(self, e_seq):
        T = e_seq.shape[0]
        s_seq = []
        for t in range(T):
            s_seq.append(self.deconv(e_seq[t]))
        s_seq = torch.stack(s_seq)
        return s_seq

    def evaluate(self, video_i, video_n):
        T = video_i.shape[1]
        n = video_i.shape[0]

        video_i = torch.transpose(video_i, dim0=0, dim1=1)  # T x n x c x h x w
        video_n = torch.transpose(video_n, dim0=0, dim1=1)  # T x n x c x h x w
        video = torch.cat([video_i, video_n], dim=1)  # T x 2n x c x h x w

        with torch.no_grad():
            s_seq = self._encode_video(video)  # T x 2n x s

        e_seq = self._encode(s_seq)  # T x 2n x z
        h_seq, hidden = self.lstm_enc(e_seq)  # T x 2n x z
        s0_seq = self._decode(e_seq)

        t, c_t, nc_t = utils.context_indices(T, context_width=2)

        e_t = e_seq[t]
        e_c_t = e_seq[c_t]
        e_nc_t = e_seq[nc_t]

        h_t = h_seq[t]
        h_c_t = h_seq[c_t]
        h_nc_t = h_seq[nc_t]

        h_i_seq = h_seq[:, :n, :]
        h_n_seq = h_seq[:, n:, :]

        l_sns = 0.
        for i in range(T):
            l_sns += self.loss_sns(h_i_seq[i], h_i_seq[i][list(range(1, n)) + [0]], h_n_seq[i])
        l_sns /= T
        l_frame = self.loss_sns(e_t, e_c_t, e_nc_t) + self.contrast_loss(torch.stack([e_t, e_c_t], dim=1))
        l_seq = self.loss_sns(h_t, h_c_t, h_nc_t)
        l_vaes = self.loss_vae(s_seq[:t+1], self._decode(self.lstm_dec(h_t, t+1)))
        l_vaei = self.loss_vae(s_seq, s0_seq)

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
            'l_vaei': l_vaei.item()
        }

        return metrics, loss

    def encode_frame(self, image):
        shape = image.shape
        if len(shape) == 3:
            image = image.unsqueeze(0)  # 1 x c x h x w
        e = self.img_encoder(image)
        e = self.conv(e)
        if len(shape) == 3:
            e = e.squeeze()
        return e
