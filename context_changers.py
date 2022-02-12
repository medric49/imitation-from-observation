import random


class ReacherHardContextChanger:

    def __init__(self):
        self.reset()

    def reset(self):
        # self.ground_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.ground_color = [1., 1., 1., 1]

        self.target_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.arm_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.root_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]
        self.finger_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, 1]

        self.c1_pos = [random.random() * 0.6 - 0.3, random.random() * 0.6 - 0.3, 2.5e-6]
        self.c2_pos = [random.random() * 0.6 - 0.3, random.random() * 0.6 - 0.3, 2.5e-6]
        self.c3_pos = [random.random() * 0.6 - 0.3, random.random() * 0.6 - 0.3, 2.5e-6]
        self.c4_pos = [random.random() * 0.6 - 0.3, random.random() * 0.6 - 0.3, 2.5e-6]
        self.c5_pos = [random.random() * 0.6 - 0.3, random.random() * 0.6 - 0.3, 2.5e-6]

        self.c1_visible = round(random.random())
        self.c2_visible = round(random.random())
        self.c3_visible = round(random.random())
        self.c4_visible = round(random.random())
        self.c5_visible = round(random.random())

        self.c1_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, self.c1_visible]
        self.c2_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, self.c2_visible]
        self.c3_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, self.c3_visible]
        self.c4_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, self.c4_visible]
        self.c5_color = [random.random() * 0.5, random.random() * 0.5, random.random() * 0.5, self.c5_visible]

    def change_env(self, env):
        env.physics.named.model.mat_texid['grid'] = -1
        env.physics.named.model.geom_rgba['ground'] = self.ground_color
        env.physics.named.model.geom_rgba['target'] = self.target_color
        env.physics.named.model.geom_rgba['arm'] = self.arm_color
        env.physics.named.model.geom_rgba['hand'] = self.arm_color
        env.physics.named.model.geom_rgba['root'] = self.root_color
        env.physics.named.model.geom_rgba['finger'] = self.finger_color

        env.physics.named.data.geom_xpos['c1'] = self.c1_pos
        env.physics.named.data.geom_xpos['c2'] = self.c2_pos
        env.physics.named.data.geom_xpos['c3'] = self.c3_pos
        env.physics.named.data.geom_xpos['c4'] = self.c4_pos
        env.physics.named.data.geom_xpos['c5'] = self.c5_pos

        env.physics.named.model.geom_rgba['c1'] = self.c1_color
        env.physics.named.model.geom_rgba['c2'] = self.c2_color
        env.physics.named.model.geom_rgba['c3'] = self.c3_color
        env.physics.named.model.geom_rgba['c4'] = self.c4_color
        env.physics.named.model.geom_rgba['c5'] = self.c5_color


    def reset_env(self, env):
        env.physics.named.model.mat_texid['grid'] = 1
        env.physics.named.model.geom_rgba['ground'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['target'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['arm'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['hand'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['root'] = [0.5, 0.5, 0.5, 1]
        env.physics.named.model.geom_rgba['finger'] = [0.5, 0.5, 0.5, 1]

        env.physics.named.data.geom_xpos['c1'] = [0, 0, 2.5e-6]
        env.physics.named.data.geom_xpos['c2'] = [0, 0, 2.5e-6]
        env.physics.named.data.geom_xpos['c3'] = [0, 0, 2.5e-6]
        env.physics.named.data.geom_xpos['c4'] = [0, 0, 2.5e-6]
        env.physics.named.data.geom_xpos['c5'] = [0, 0, 2.5e-6]

        env.physics.named.model.geom_rgba['c1'] = [0, 0, 0, 0]
        env.physics.named.model.geom_rgba['c2'] = [0, 0, 0, 0]
        env.physics.named.model.geom_rgba['c3'] = [0, 0, 0, 0]
        env.physics.named.model.geom_rgba['c4'] = [0, 0, 0, 0]
        env.physics.named.model.geom_rgba['c5'] = [0, 0, 0, 0]