_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
dataset_type = 'CocoDataset'
data_root = 'data/chicken/'
classes = ('none','Chicken',)
model = dict(
roi_head=dict(
bbox_head=dict(
    num_classes=2
),
mask_head=dict(
    num_classes=2
)))
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/_annotations.coco.json',
        img_prefix=data_root + 'train/',
        ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'valid/_annotations.coco.json',
        img_prefix=data_root + 'valid/',
        ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test/_annotations.coco.json',
        img_prefix=data_root + 'test/',
        )
   
  )
optimizer = dict(type='SGD', lr=0.06, momentum=0.9, weight_decay=5e-4)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[4, 7])


runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=10)

log_config = dict(
    interval=1,
    )
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
work_dir = './work_dirs/chicken'
