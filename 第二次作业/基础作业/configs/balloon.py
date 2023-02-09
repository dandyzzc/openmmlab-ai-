_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
dataset_type = 'CocoDataset'
data_root = 'data/balloon/'
classes = ('balloon',)
model = dict(
roi_head=dict(
bbox_head=dict(
    num_classes=1
),
mask_head=dict(
    num_classes=1
)))
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'ann/train_ann.json',
        img_prefix=data_root + 'train/',
        ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'ann/val_ann.json',
        img_prefix=data_root + 'val/',
        ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'ann/val_ann.json',
        img_prefix=data_root + 'val/',
        )
   
  )
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=4e-6)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[8, 11])


runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=25)

log_config = dict(
    interval=1,
    )
load_from = './work_dirs/balloon3/epoch_50.pth'
work_dir = './work_dirs/balloon'
