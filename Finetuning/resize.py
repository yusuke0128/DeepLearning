import cv2
import csv
import pandas as pd
import numpy as np
datasetURL = '/home/yusuke/dataset/style-color-images/style/'
outDatasetURL = '/home/yusuke/dataset/style-color-images/style2/'
dataFl = pd.read_csv('/home/yusuke/dataset/style-color-images/style.csv')
fileNameSet = np.array(dataFl['file'].values,dtype=str)

for i in range(0,len(fileNameSet)):
	img = cv2.imread(datasetURL+fileNameSet[i])
	img = cv2.resize(img,(244,244))
	cv2.imwrite(outDatasetURL+fileNameSet[i],img)
