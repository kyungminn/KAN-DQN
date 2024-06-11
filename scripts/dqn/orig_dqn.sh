cd ..
cd ..
python dqn.py \
    --config_name kan_dqn \
    --overrides group_name='atari' \
    --overrides exp_name='orig_DQN' \
    --overrides wandb_run_name='lr5e4_rr1_seed2_epi1e5' \
    --overrides seed=2 \
    --overrides buffer_limit=100000 \
    --overrides num_episodes_per_env=100000 \
    --overrides num_eval_envs=10