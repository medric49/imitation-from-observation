import torch
from torch import nn


class CTEncNet(nn.Module):
    def __init__(self, hidden_dim):
        super(CTEncNet, self).__init__()
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
        z = self.flatten(self.fc2(z))
        return z, c1, c2, c3, c4


class CTDecNet(nn.Module):
    def __init__(self, hidden_dim):
        super(CTDecNet, self).__init__()
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
        obs = self.leaky_relu(self.b_norm_fc(self.fc(z)))
        obs = self.leaky_relu(self.conn_4(torch.cat([c4, obs], dim=1)))
        obs = self.leaky_relu(self.b_norm_4(self.t_conv_4(obs)))
        obs = self.leaky_relu(self.conn_3(torch.cat([c3, obs], dim=1)))
        obs = self.leaky_relu(self.b_norm_3(self.t_conv_3(obs)))
        obs = self.leaky_relu(self.conn_2(torch.cat([c2, obs], dim=1)))
        obs = self.leaky_relu(self.b_norm_2(self.t_conv_2(obs)))
        obs = self.leaky_relu(self.conn_1(torch.cat([c1, obs], dim=1)))
        obs = self.t_conv_1(obs)
        return obs


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



