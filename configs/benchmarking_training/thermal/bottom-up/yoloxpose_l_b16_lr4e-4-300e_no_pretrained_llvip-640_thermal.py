_base_ = "yoloxpose_s_b16_lr4e-4-300e_llvip-640_thermal.py"
work_dir = "runs/benchmarks/thermal/yolopose-l_b16_lr4e-4_no_pretrained_backbone_thermal"

widen_factor = 1
deepen_factor = 1
checkpoint = None

# model settings
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=checkpoint,
    ),
    neck=dict(in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    head=dict(head_module_cfg=dict(widen_factor=widen_factor)),
)
