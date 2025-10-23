_base_ = [
    "convnext-large_upernet_2xb20-amp-80k_wildscenes-512x512_template.py",
    "../_base_/datasets/wildscenes_standard.py"
]

num_classes = _base_.num_classes
crop_size = _base_.crop_size

model = dict(
    decode_head=dict(
        type='EDLUPerHead',
        evidence_func='exp',
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='EDLSegLoss',
            num_classes=num_classes,
            annealing_steps=10000,
            kl_weight=0.2,
            loss_weight=1.0,
            ignore_index=255,
            loss_name='loss_edl')
    ),
    auxiliary_head=dict(
        type='EDLUPerHead',
        evidence_func='exp',
        in_channels=768,        # ← 单层输入
        in_index=2,             # ← 单层输入
        pool_scales=(1, 2, 3, 6),
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='EDLSegLoss',
            num_classes=num_classes,
            annealing_steps=10000,
            kl_weight=0.2,
            loss_weight=0.4,
            ignore_index=255,
            loss_name='loss_edl_aux')
    )
)