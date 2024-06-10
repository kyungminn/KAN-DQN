cd ..
cd ..
python dqn.py \
    --config_name kan_dqn \
    --overrides group_name='atari' \
    --overrides exp_name='KAN_DQN_seed0' \
    --overrides wandb_run_name='lr5e4_rr1_seed0_epi1e6' \
    --overrides model='KAN_DQN' \
    --overrides seed=0 \
    --overrides buffer_limit=1000000 \
    --overrides num_episodes_per_env=1000000 \
    --overrides num_eval_envs=10