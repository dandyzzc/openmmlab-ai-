_base_ = [
    '../_base_/models/resnet50_cifar.py', '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]
# dataset_type = 'CustomDataset'
data = dict(
    samples_per_gpu = 64, # 单卡 batchsize
    workers_per_gpu=2,
    )
model = dict(
    head=dict(
        topk=(1, ),        # top-k
    ))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=4e-5)
lr_config = dict(policy='step', step=[4, 8])
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=1,
    metric=['accuracy', 'precision', 'f1_score'],
    metric_options=dict(topk=(1, )))

load_from = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth'