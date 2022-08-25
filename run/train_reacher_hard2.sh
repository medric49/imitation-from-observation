# python3 scripts/generate_dmc_video.py --env reacher_hard2 --episode_len 60

python3 train_ct.py task=reacher_hard2 episode_len=60 exp_id=1 batch_size=16

python3 train_rl.py task=reacher_hard2 episode_len=60