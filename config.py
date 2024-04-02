# model settings
import argparse

model = dict(
    type='BMN',
    temporal_dim=100,
    boundary_ratio=0.5,
    num_samples=32,
    num_samples_per_bin=3,
    feat_dim=400,
    soft_nms_alpha=0.4,
    soft_nms_low_threshold=0.5,
    soft_nms_high_threshold=0.9,
    post_process_top_k=100)

# data processing
pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(224, 224)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False),
        dict(type='FormatShape', input_format='NCHW'),
    ]

# optical_flow_raft_model_settings
optical_flow_raft_model = dict(
    type='RAFT',
    args=argparse.Namespace(
        small=False,
        mixed_precision=True,
        alternate_corr=False,
    ))

# dataset settings
dataset_type = 'ActivityNetDataset'
data_root = 'data/activitynet_feature_cuhk/'
anno_root = 'data/activitynet_annotations/'

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=1),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=1, test_mode=True),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=3,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=anno_root + 'anet_anno_train.json',
        data_prefix=data_root + 'train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=anno_root + 'anet_anno_val.json',
        data_prefix=data_root + 'val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=anno_root + 'anet_anno_test.json',
        data_prefix=data_root + 'test',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[1, 2], gamma=0.1)
total_epochs = 3

# checkpoint saving
checkpoint_config = dict(interval=1)

# log config
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/bmn'
load_from = None
resume_from = None
workflow = [('train', 1)]
