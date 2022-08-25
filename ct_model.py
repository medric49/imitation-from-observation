from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import nets


class CTModel(nn.Module):
    def __init__(self, hidden_dim, lr):
        super(CTModel, self).__init__()

        self.enc1 = nets.CTEncNet(hidden_dim)
        self.enc2 = nets.CTEncNet(hidden_dim)
        self.t = nets.TranslatorNet(hidden_dim)
        # self.t = nets.LSTMTranslatorNet(hidden_dim)
        self.dec = nets.CTDecNet(hidden_dim)

        self._enc1_opt = torch.optim.Adam(self.enc1.parameters(), lr=lr)
        self._enc2_opt = torch.optim.Adam(self.enc2.parameters(), lr=lr)
        self._t_opt = torch.optim.Adam(self.t.parameters(), lr=lr)
        self._dec_opt = torch.optim.Adam(self.dec.parameters(), lr=lr)
    
    def translate(self, video1, fobs2):
        T = video1.shape[0]

        video1 = video1.to(dtype=torch.float) / 255.  # T x c x h x w
        fobs2 = fobs2.to(dtype=torch.float) / 255.  # c x h x w
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

    def encode_frame(self, image):
        shape = image.shape
        if len(shape) == 3:
            image = image.unsqueeze(0)  # 1 x c x h x w
        image = image.to(dtype=torch.float) / 255.
        e = self.enc1(image)[0]
        if len(shape) == 3:
            e = e.squeeze()
        return e

    @staticmethod
    def load(file):
        snapshot = Path(file)
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        return payload['context_translator']
