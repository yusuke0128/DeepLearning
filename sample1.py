import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import sys

plt.style.use('ggplot')
# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
batchsize = 100

# 学習の繰り返し回数
n_epoch   = 20

# 中間層の数
n_units   = 1000
mnist = fetch_mldata('MNIST original')
mnist.data = mnist.data.astype(np.float32)
mnist.data /=255

mnist.target = mnist.target.astype(np.int32)

def draw_digit(data):
	size=28
	plt.figure(figsize=(2.5,3))
	X,Y = np.meshgrid(range(size),range(size))
	Z = data.reshape(size,size)
	Z = Z[::-1,:]
	plt.xlim(0,27)
	plt.ylim(0,27)
	plt.pcolor(X, Y, Z)
	plt.gray()
	plt.tick_params(labelbottom="off")
	plt.tick_params(labelleft="off")
	plt.show()

#draw_digit(mnist.data[5])
#draw_digit(mnist.data[12345])
#draw_digit(mnist.data[33456])

# 学習用データを N個、検証用データを残りの個数と設定
N = 60000
x_train, x_test = np.split(mnist.data,   [N])
y_train, y_test = np.split(mnist.target, [N])
N_test = y_test.size

model = FunctionSet(l1=F.Linear(784, n_units),
                    l2=F.Linear(n_units, n_units),
                    l3=F.Linear(n_units, 10)).to_gpu()

def forward(x_data, y_data, train=True):
	x,t = Variable(x_data),Variable(y_data)
	h1 = F.dropout(F.relu(model.l1(x)), train=train)
	h2 = F.dropout(F.relu(model.l2(h1)), train=train)
	y = model.l3(h2)
	return F.softmax_cross_entropy(y,t), F.accuracy(y,t)

optimizer = optimizers.Adam()
optimizer.setup(model)

train_loss = []
train_acc  = []
test_loss = []
test_acc  = []

l1_W = []
l2_W = []
l3_W = []

# Learning loop
for epoch in range(1, n_epoch+1):
    print('epoch', epoch)

    # training
    # N個の順番をランダムに並び替える
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    # 0〜Nまでのデータをバッチサイズごとに使って学習
    for i in range(0, N, batchsize):
        x_batch =cuda.to_gpu(x_train[perm[i:i+batchsize]])
        y_batch =cuda.to_gpu(y_train[perm[i:i+batchsize]])
        # 勾配を初期化
        optimizer.zero_grads()
        # 順伝播させて誤差と精度を算出
        loss, acc = forward(x_batch, y_batch)
        # 誤差逆伝播で勾配を計算
        loss.backward()
        optimizer.update()
        train_loss.append(loss.data)
        train_acc.append(acc.data)
        sum_loss     += float(cuda.to_gpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_gpu(acc.data)) * batchsize

    # 訓練データの誤差と、正解精度を表示
    print('train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N))

    # evaluation
    # テストデータで誤差と、正解精度を算出し汎化性能を確認
    sum_accuracy = 0
    sum_loss     = 0
    for i in range(0, N_test, batchsize):
        x_batch =cuda.to_gpu( x_test[i:i+batchsize])
        y_batch =cuda.to_gpu( y_test[i:i+batchsize])

        # 順伝播させて誤差と精度を算出
        loss, acc = forward(x_batch, y_batch, train = False)
        test_loss.append(loss.data)
        test_acc.append(acc.data)
        sum_loss     += float(cuda.to_gpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_gpu(acc.data)) * batchsize

    # テストデータでの誤差と、正解精度を表示
    print ('test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test))

    # 学習したパラメーターを保存
    l1_W.append(model.l1.W)
    l2_W.append(model.l2.W)
    l3_W.append(model.l3.W)
train_acc = np.array(train_acc,np.float32)
test_acc = np.array(test_acc,np.float32)
print(train_acc)
# 精度と誤差をグラフ描画
plt.figure(figsize=(8,6))
plt.plot(range(train_acc.size), train_acc)
plt.plot(range(test_acc.size), test_acc)
plt.legend(["train_acc","test_acc"],loc=4)
plt.title("Accuracy of digit recognition.")
plt.plot()
plt.show()
