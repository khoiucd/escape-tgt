log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=10, metric='PCK', key_indicator='PCK', gpu_collect=True)
optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[40, 50])
total_epochs = 60
log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=1115,
    dataset_joints=1,
    dataset_channel=[
        [
            0,
        ],
    ],
    inference_channel=[
        0,
    ],
    max_kpt_num=68)

# model settings
model = dict(
    type='PrototypeDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', depth=50),
    deconv=dict(type='CustomDeconv', in_channels=2048),
    keypoint_head=dict(
        type='GlobalHead',
        keypoints_num=channel_cfg['num_output_channels'],
        out_channels=256,
        superkeypoints=dict(
            checkpoint='work_dirs/baseline_split2/superkeypoints.pth',
            mname='superkeypoints'),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    keypoint_adaptation=dict(
        type='MLEHead',
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        pooling_kernel=15,
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11,
        fewshot_testing=True))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=15,
        scale_factor=0.15),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'category_id'
        ]),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs', 'category_id'
        ]),
]

test_pipeline = valid_pipeline

data_root = 'data/mp100'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=12,
    episodes_per_gpu=1,
    train=dict(
        type='GlobalDataset',
        ann_file=f'{data_root}/annotations/mp100_split2_train.json',
        img_prefix=f'{data_root}/images/',
        superkeypoints='work_dirs/baseline_split2/superkeypoints.pth',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        pipeline=train_pipeline),
    val=dict(
        type='FewshotDataset',
        ann_file=f'{data_root}/annotations/mp100_split2_val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        num_shots=1,
        num_queries=15,
        num_episodes=100,
        max_kpt_num=channel_cfg['max_kpt_num'],
        pipeline=valid_pipeline),
    test=dict(
        type='FewshotDataset',
        ann_file=f'{data_root}/annotations/mp100_split2_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        num_shots=1,
        num_queries=15,
        num_episodes=200,
        max_kpt_num=channel_cfg['max_kpt_num'],
        pipeline=valid_pipeline),
    extract_features=dict(
        type='GlobalDataset',
        ann_file=f'{data_root}/annotations/mp100_split2_train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        pipeline=valid_pipeline),
)

shuffle_cfg = dict(interval=1)
