import chainer
import chainer.functions as F
import chainer.links as L
from chainer.serializers import npz
class VGG16(chainer.Chain):

	def __init__(self,classLabels=7,pretrainedModel= '/home/yusuke/github/chainerModel/vgg16.npz'):
		super(VGG16,self).__init__()
		with self.init_scope():
			self.base = BaseVGG16()
			self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
			self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
			self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)
			self.fc6 = L.Linear(None,512)
			self.fc7 = L.Linear(None,128)
			self.fc8 = L.Linear(None,classLabels)
		npz.load_npz(pretrainedModel,self.base)

	def __call__(self,x,t):
		h = self.predict(x)
		loss = F.softmax_cross_entropy(h, t)
		chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
		return loss

	def predict(self,x):
		h = self.base(x)
		h = F.relu(self.conv5_1(h))
		h = F.relu(self.conv5_2(h))
		h = F.relu(self.conv5_3(h))
		h = F.max_pooling_2d(h, ksize=2, stride=2)
		h = F.dropout(F.relu(self.fc6(h)))
		h = F.dropout(F.relu(self.fc7(h)))
		return self.fc8(h)

class BaseVGG16(chainer.Chain):

	def __init__(self):
		super(BaseVGG16,self).__init__()
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

		return h

