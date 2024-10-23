_base_ = ["mmpose::_base_/default_runtime.py"]
work_dir = "runs/benchmarks/thermal/ViTPose-h_b16_lr5e-4_256x192_thermal"

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
custom_imports = dict(imports=["mmpose.engine.optim_wrappers.layer_decay_optim_wrapper"], allow_failed_imports=False)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=32,
        layer_decay_rate=0.85,
        custom_keys={
            "bias": dict(decay_multi=0.0),
            "pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        },
    ),
    constructor="LayerDecayOptimWrapperConstructor",
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])

# learning policy
param_scheduler = [
    dict(type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False),  # warm-up
    dict(type="MultiStepLR", begin=0, end=210, milestones=[170, 200], gamma=0.1, by_epoch=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best="coco/AP", rule="greater", interval=1, max_keep_ckpts=3))

# codec settings
codec = dict(type="UDPHeatmap", input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True
    ),
    backbone=dict(
        type="mmpretrain.VisionTransformer",
        arch="huge",
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.55,
        with_cls_token=False,
        out_type="featmap",
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmpose/"
            "v1/pretrained_models/mae_pretrain_vit_huge_20230913.pth",
        ),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=1280,
        out_channels=17,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode="heatmap",
        shift_heatmap=False,
    ),
)

# base dataset settings
dataset_type = "LLVIPDataset"
data_mode = "topdown"
data_root = "data/llvip/"

# pipelines
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform"),
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="PackPoseInputs"),
]

# data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/train_llvip.json",
        data_prefix=dict(img="infrared/train/"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/test_llvip.json",
        # bbox_file=data_root + "annotations/test_bbox_thermal_nms_50_llvip.json",
        data_prefix=dict(img="infrared/test/"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(type="CocoMetric", ann_file=data_root + "annotations/test_llvip.json", iou_type ="keypoints")
test_evaluator = val_evaluator
