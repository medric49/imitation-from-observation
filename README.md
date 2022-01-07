# sharingan

This project aims to train an agent to imitate an expert performing a task and observed through a video.
The first step is to train an agent using a Deep RL algorithm. 

We use the algorithm [DrQv2](https://arxiv.org/abs/2107.09645) to train the expert in a continuous control domain of [DM Control](https://github.com/deepmind/dm_control),
which allow us to build a dataset of video that is used to train an imitator agent with [Imitation from Observation](https://arxiv.org/abs/1707.03374) algorithm.

<p align="center">
<img src="demo/demo-expert.gif" width="500">
<br>
Expert's training in DM Control's Manipulator domain
</p>

## Installation

* Install the environment
```shell
conda env create -f env.yml
conda activate sharingan
```

* Train the expert
```shell
python train.py
```

* Watch training evolution in Tensorboard
```shell
tensorboard --logdir exp_local
```

## Acknowledgements
This code is inspired by
* Denis Yarats's [DrQv2 project](https://github.com/facebookresearch/drqv2)

## References
* [Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning](https://arxiv.org/pdf/2107.09645.pdf)
* [Imitation from Observation: Learning to Imitate Behaviors from Raw Video via Context Translation](https://arxiv.org/pdf/1707.03374.pdf)
