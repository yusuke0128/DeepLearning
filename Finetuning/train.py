import chainer
import chainer.functions as F
import chainer.links as L
import dataset
from chainer.serializers import npz
from chainer import iterators
from chainer import training
from chainer.training import extensions
import numpy as np
from chainer.cuda import to_cpu
import random

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

class VGG(chainer.Chain):

	def __init__(self,classLabels=7,pretrainedModel= '/home/yusuke/github/chainerModel/vgg16.npz'):
		super(VGG,self).__init__()
		with self.init_scope():
			self.base = BaseVGG
			self.fc6 = L.Linear(None,512)
			self.fc7 = L.Linear(None,128)
			self.fc8 = L.Linear(None,classLabels)
		npz.load_npz(pretrainedModel,self.base)

	def __call__(self,x,t):
		h = self.predict(x)
		loss = F.softmax_cross_entropy(h, t)
		return loss

	def predict(self,x):
		h = self.base(x)
		h = F.dropout(F.relu(self.fc6(h)))
		h = F.dropout(F.relu(self.fc7(h)))
		return self.fc8(h)

class BaseVGG(chainer.Chain):

	def __init__(self):
		super(BaseVGG,self).__init__()
		with self.init_scope():
			self.conv1_1 = L.Convolution2D(None, 64, 3, 1, 1)
			self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)
			self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
			self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)
			self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
			self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
			self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)
			self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
			self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
			self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)
			self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
			self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
			self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)

	def __call__(self,x):
		h = F.relu(self.conv1_1(x))
		h = F.relu(self.conv1_2(h))
		h = F.max_pooling_2d(h, ksize=2, stride=2)

		h = F.relu(self.conv2_1(h))
		h = F.relu(self.conv2_2(h))
		h = F.max_pooling_2d(h, ksize=2, stride=2)

		h = F.relu(self.conv3_1(h))
		h = F.relu(self.conv3_2(h))
		h = F.relu(self.conv3_3(h))
		h = F.max_pooling_2d(h, ksize=2, stride=2)

		h = F.relu(self.conv4_1(h))
		h = F.relu(self.conv4_2(h))
		h = F.relu(self.conv4_3(h))
		h = F.max_pooling_2d(h, ksize=2, stride=2)

		h = F.relu(self.conv5_1(h))
		h = F.relu(self.conv5_2(h))
		h = F.relu(self.conv5_3(h))
		h = F.max_pooling_2d(h, ksize=2, stride=2)

		return h

model = VGG()

if gpuId >= 0:
    model.to_gpu(gpuId)

optimizer = chainer.optimizers.MomentumSGD()
optimizer.setup(model)
model.base.disable_update()
#model.conv5_1.disable_update()
#model.conv5_2.disable_update()
#model.conv5_3.disable_update()
model = L.Classifier(model)

updater = training.StandardUpdater(trainIter, optimizer, device=gpuId)
trainer = training.Trainer(updater, (maxEpoch, 'epoch'), out='result')
trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.run()

