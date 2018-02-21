#coding: utf-8
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import rdBase
print 'RDKit version ',rdBase.rdkitVersion
 
import chainer
from chainer import computational_graph as c
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
print 'Chainer version ',chainer.__version__
 
import six
import numpy as np
import sys
 
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
 
# 記述子の準備
act_classes={'(A) low':0,'(B) medium':1,'(C) high':2}
descList = [x[0] for x in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descList)
 
# 訓練データの読み込み
# 訓練データの読み込み
# データセットは、以下のサイトにある化合物のlogS値(Huuskonen dataset)を利用しました。
# 入手先：http://sourceforge.net/p/rdkit/code/HEAD/tree/trunk/Docs/Book/data/
# ファイル名: solubility.train.sdf, solubility.test.sdf
#  solubility.train.sdfに含まれるCyhexatinは、一部の記述子が計算できなかったので、除外しています。
m_train = [m for m in Chem.SDMolSupplier('solubility.train.sdf') if m is not None]
x_train = [calc.CalcDescriptors(m) for m in m_train]
x_train = np.array(x_train,dtype=np.float32)
y_train = [act_classes[m.GetProp('SOL_classification')] for m in m_train]
y_train = np.array(y_train,dtype=np.int32)
 
# テストデータの読み込み
m_test = [m for m in Chem.SDMolSupplier('solubility.test.sdf') if m is not None]
x_test = [calc.CalcDescriptors(m) for m in m_test]
x_test = np.array(x_test,dtype=np.float32)
y_test = [act_classes[m.GetProp('SOL_classification')] for m in m_test]
y_test = np.array(y_test,dtype=np.int32)
 
# 訓練データ、テストデータのスケーリング
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)
 
# パラメータ設定
batchsize = 100         # 確率的勾配降下法で学習させるバッチサイズ
n_epoch = 30            # 学習の繰り返し回数
n_units = 1000          # 中間層のユニット数
Ndesc = len(descList)   # 入力層のユニット数(記述子の数)
Nout = 3                # 出力層のユニット数(クラス数)
N = y_train.size        # 訓練データ数
N_test = y_test.size    # テストデータ数
 
 
'''元コード
# Prepare multi-layer perceptron model
# 入力：記述子の数(196)　出力：クラスの数(3)
model = chainer.FunctionSet(l1=F.Linear(Ndesc, n_units),
                            l2=F.Linear(n_units, n_units),
                            l3=F.Linear(n_units, Nout))
 
# Neural net architecture
def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y = model.l3(h2)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
'''

model_param = dict(N_in = Ndesc,
				   N_unit = n_units,
				   N_out = Nout)


class MyChain(Chain):
	def __init__(self,model_param):
		super(MyChain, self).__init__(
			l1 = L.Linear(model_param['N_in'],model_param['N_unit']),
			l2 = L.Linear(model_param['N_unit'], model_param['N_unit']),
			l3 = L.Linear(model_param['N_unit'], model_param['N_out']),
		)
	def __call__(self,x,t):
		x, t =  chainer.Variable(np.array(x)), chainer.Variable(np.array(t))
		y = self.fwd(x)
		return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
	
	def fwd(self,x):
		#h1 = F.dropout(F.relu(self.l1(x)),  train=True)
		#h2 = F.dropout(F.relu(self.l2(h1)), train=True)
		h1 = F.relu(self.l1(x))
		h2 = F.relu(self.l2(h1))
		y = self.l3(h2)
		return y	

		
			

# Setup optimizer
model = MyChain(model_param)
optimizer = optimizers.Adam()
optimizer.setup(model)
 
# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
 
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
 
    for i in six.moves.range(0, N, batchsize):
        x_batch = x_train[perm[i:i + batchsize]]
        y_batch = y_train[perm[i:i + batchsize]]
        model.zerograds()
        loss, acc = model(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)
 
    print('epoch={}, train mean loss={}, accuracy={}'.format(
        epoch, sum_loss / N, sum_accuracy / N))
 
# テストデータのクラス予測
x, t = chainer.Variable(np.array(x_test)), chainer.Variable(np.array(y_test))
y = model.fwd(x)
loss = F.softmax_cross_entropy(y, t)
acc = F.accuracy(y, t)
print('test  mean loss={}, accuracy={}'.format(loss.data, acc.data))
 
# テストデータの評価 (Confusion matrixの計算)
prob = F.softmax(y).data
pred_y = []
for p,y in zip(prob,y_test):
    pre_y= np.argmax(p)
    #print pre_y,y
    pred_y.append(pre_y)
pred_y = np.array(pred_y)
confmat = metrics.confusion_matrix(y_test,pred_y)
print confmat
