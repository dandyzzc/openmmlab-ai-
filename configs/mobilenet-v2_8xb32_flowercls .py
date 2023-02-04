_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]
dataset_type = 'CustomDataset'
model = dict(
    head=dict(
        num_classes=5,     # 分类个数
        topk=(1, ),        # top-k
    ))

## 数据集
data = dict(
    samples_per_gpu = 32, # 单卡 batchsize
    workers_per_gpu=2,
    # 训练集
    train=dict(
        type=dataset_type,
        data_prefix='data/flower_split/train',
        ann_file='data/flower_split/train.txt',
        classes='data/flower_split/classes.txt',),
        
    val=dict(
        type=dataset_type,
        data_prefix='data/flower_split/val',
        ann_file='data/flower_split/val.txt',
        classes='data/flower_split/classes.txt',),
    )
## 评估指标
evaluation = dict(interval=1, metric=['accuracy','precision','f1_score'], metric_options=dict(topk=(1,)))

## 学习率与优化器
optimizer = dict(type='SGD', lr=0.0012, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# 学习率策略
lr_config = dict(
    policy='step',
    step=[4,8])
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=5)
## 预训练模型
# load_from = None # 随机初始化
# load_from = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
load_from = '/root/mmclassification/work_dirs/mobilenet-v2_8xb32_flowercls/best.pth'