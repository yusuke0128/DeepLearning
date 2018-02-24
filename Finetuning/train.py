import chainer
import chainer.functions as F
import chainer.links as L
import dataset
from chainer import iterators
from chainer import training
from chainer.training import extensions
import numpy as np
from chainer.cuda import to_cpu
import random
import VGG16

random.seed(0)
np.random.seed(0)

if chainer.cuda.available:
    chainer.cuda.cupy.random.seed(0)

gpuId = 0

txtFileURL = 'dataset.txt'
Dataset = dataset.Dataset()
dataset = Dataset.readDataset(txtFileURL)
dataset = chainer.datasets.TransformDataset(dataset, Dataset.normarize)
batchsize = 100
maxEpoch = 100
splitAt = int(500)
trainDataset,testDataset = chainer.datasets.split_dataset_random(dataset,splitAt)
print(trainDataset[499])
trainIter = iterators.SerialIterator(trainDataset, batchsize)

model = VGG16.VGG16()

if gpuId >= 0:
    model.to_gpu(gpuId)

optimizer = chainer.optimizers.MomentumSGD()
optimizer.setup(model)
model.base.disable_update()
model = L.Classifier(model)

updater = training.StandardUpdater(trainIter, optimizer, device=gpuId)
trainer = training.Trainer(updater, (maxEpoch, 'epoch'), out='/home/yusuke/dataset/style-color-images/result')
trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.run()
