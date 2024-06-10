cd ..
cd ..
python dqn.py \
    --config_name kan_dqn \
    --overrides group_name='atari' \
    --overrides exp_name='orig_DQN_seed2' \
    --overrides wandb_run_name='lr5e4_rr1_seed2_epi1e6' \
    --overrides seed=2 \
    --overrides buffer_limit=1000000 \
    --overrides num_episodes_per_env=1000000 \
    --overrides num_eval_envs=10