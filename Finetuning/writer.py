import csv
import pandas as pd
import numpy as np
datasetURL = '/home/yusuke/dataset/style-color-images/style2/'
dataFl = pd.read_csv('/home/yusuke/dataset/style-color-images/style.csv')
fileNameSet = np.array(dataFl['file'].values,dtype=str)
brandLabelSet = np.array(dataFl['brand_label'].values,dtype=str)
print(len(brandLabelSet))
print(fileNameSet[0])
f = open('dataset.txt','w')

for i in range(0,len(fileNameSet)):
	f.write(datasetURL+fileNameSet[i]+' '+brandLabelSet[i]+'\n')
f.close

