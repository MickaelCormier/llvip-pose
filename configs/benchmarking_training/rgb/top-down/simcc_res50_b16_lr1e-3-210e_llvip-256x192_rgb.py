_base_ = ["mmpose::_base_/default_runtime.py"]
work_dir = "runs/benchmarks/rgb/simcc_res50_b16_lr1e-3_256x192_rgb"

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)
# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type="Adam",
        lr=1e-3,
    )
)

default_hooks = dict(checkpoint=dict(save_best="coco/AP", rule="greater", interval=1, max_keep_ckpts=3))

# learning policy
param_scheduler = [
    dict(type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False),  # warm-up
    dict(type="MultiStepLR", milestones=[170, 200], gamma=0.1, by_epoch=True),
]

visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(type="SimCCLabel", input_size=(192, 256), sigma=6.0, simcc_split_ratio=2.0)

# model settings
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    head=dict(
        type="SimCCHead",
        in_channels=2048,
        out_channels=17,
        input_size=codec["input_size"],
        in_featuremap_size=tuple(s // 32 for s in codec["input_size"]),
        simcc_split_ratio=codec["simcc_split_ratio"],
        loss=dict(type="KLDiscretLoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True),
)

# base dataset settings
dataset_type = "LLVIPDataset"
data_mode = "topdown"
data_root = "data/llvip"

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
        data_prefix=dict(img="visible/"),
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
        # bbox_file=data_root + "annotations/test_bbox_rgb_nms_50_llvip.json",
        data_prefix=dict(img="visible/"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(type="CocoMetric", ann_file=data_root + "annotations/test_llvip.json", iou_type ="keypoints")
test_evaluator = val_evaluator