simple_cfg = {
    'lr': 0.001,
    'lr_step_size': 100,
    'lr_gamma': 1,
    'ctx_win_len': 40,
    'cond_win_len': 20,
    'batch_size': 64,
    'num_covariates': 3,  # wday, hour, min
    'total_num_covariates': 12,
    'num_time_idx': 1,
    'num_targets': 1,
    'total_num_targets': 1,
    'num_lstms': 2,
    'hidden_dim': 64,
    'max_batches_per_epoch': 400,
    'num_epochs': 50,  # converges in about 50 iterations
    'data_file': 'SML2010.csv',
    'train_test_split': 0.9 # don't have much data in this dataset so use more data for training
}

full_cfg = simple_cfg
full_cfg['num_covariates'] =12
