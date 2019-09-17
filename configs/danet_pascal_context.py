model:
    arch: danet_50
data:
    dataset: pcontext
    train_split: train
    val_split: val
    # img_rows: 1024
    # img_cols: 2048
    path: datasets/downloads/
training:
    train_iters: 29750
    batch_size: 2
    val_interval: 100
    print_interval: 50
    n_workers: 16
    optimizer:
        name: 'sgd'
        lr: 1.0e-4
        weight_decay: 0.0005
        momentum: 0.9
    lr_schedule:
    loss:
      name: 'cross_entropy'
      size_average: False
    # l_rate: 1.0e-4
    # l_schedule:
    # momentum: 0.99
    # weight_decay: 0.0005
    resume: danet_cityscapes_best_model.pkl
    ckpt_path: checkpoints/
    visdom: False
