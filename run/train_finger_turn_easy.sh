python3 scripts/generate_dmc_video.py --env finger_turn_easy --episode_len 60

python3 train_ct.py task=finger_turn_easy episode_len=60 exp_id=1 batch_size=16

python3 train_rl.py task=finger_turn_easy episode_len=60 n_video=10