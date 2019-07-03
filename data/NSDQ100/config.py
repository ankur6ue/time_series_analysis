simple_cfg = {
    'lr': 0.001,
    'lr_step_size': 100,
    'lr_gamma': 1,
    'ctx_win_len': 40,
    'cond_win_len': 20,
    'batch_size': 128,
    'num_covariates': 4,
    'total_num_covariates': 99,
    'num_time_idx': 1,
    'num_targets': 2,
    'total_num_targets': 2,
    'num_lstms': 2,
    'hidden_dim': 64,
    'max_batches_per_epoch': 400,
    'num_epochs': 10, # converges in about 5 iterations
    'data_file': '/../../data/NSDQ100/nasdaq100_padding.csv',
    'train_test_split': 0.9 # don't have much data in this dataset so use more data for training
}

full_cfg = simple_cfg
full_cfg['num_covariates'] = 70