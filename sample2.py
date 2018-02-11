
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import sys

plt.style.use('ggplot')
# dropout(x, ratio=0.5, train=True) テスト
# x: 入力値
# ratio: 0を出力する確率
# train: Falseの場合はxをそのまま返却する
# return: ratioの確率で0を、1−ratioの確率で,x*(1/(1-ratio))の値を返す

n = 50
v_sum = 0
for i in range(n):
    x_data = np.array([1,2,3,4,5,6], dtype=np.float32)
    x = Variable(x_data)
    dr = F.dropout(x, ratio=0.6,train=True)

    for j in range(6):
        sys.stdout.write( str(dr.data[j]) + ', ' )
    print("")
    v_sum += dr.data

# outputの平均がx_dataとだいたい一致する 
sys.stdout.write( str((v_sum/float(n))) )

