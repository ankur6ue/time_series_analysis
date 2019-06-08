simple_cfg = {
    'lr': 0.001,
    'lr_step_size': 100,
    'lr_gamma': 1,
    'ctx_win_len': 192,
    'cond_win_len': 168,
    'batch_size': 128,
    'num_covariates': 3,
    'total_num_covariates': 3,
    'num_time_idx': 1,
    'num_targets': 100,
    'total_num_targets': 369,
    'num_lstms': 2,
    'hidden_dim': 64,
    'max_batches_per_epoch': 400,
    'num_epochs': 100,
    'data_file': 'electricity.csv',
    'train_test_split': 0.7
}

full_cfg = simple_cfg
full_cfg['num_covariates'] = 3