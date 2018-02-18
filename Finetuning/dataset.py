import chainer
import numpy as np
class Dataset:

	def __init__(self):
		self.txtURL = ' '
		self.data = 0

	def readDataset(self,txtURL):
		imageFiles = txtURL
		dataset = chainer.datasets.LabeledImageDataset(imageFiles)
		return dataset

	def normarize(self,data):
		img,label = data
		img = img/255.
		return img,label
