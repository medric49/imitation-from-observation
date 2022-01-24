import random


class ReacherHardContextChanger:

    def __init__(self):
        self.reset()

    def reset(self):
        self.ground_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.target_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.arm_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.root_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.finger_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]

    def change_env(self, env):
        env.physics.named.model.mat_texid['grid'] = -1
        env.physics.named.model.geom_rgba['ground'] = self.ground_color
        env.physics.named.model.geom_rgba['target'] = self.target_color
        env.physics.named.model.geom_rgba['arm'] = self.arm_color
        env.physics.named.model.geom_rgba['hand'] = self.arm_color
        env.physics.named.model.geom_rgba['root'] = self.root_color
        env.physics.named.model.geom_rgba['finger'] = self.finger_color

    def reset_env(self, env):
        env.physics.named.model.mat_texid['grid'] = 1
        env.physics.named.model.geom_rgba['ground'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['target'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['arm'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['hand'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['root'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['finger'] = [0.5, 0.5, 0.5, 1]
