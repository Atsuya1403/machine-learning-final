#import
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#データセットを読み込む
ds = pd.read_csv("/Users/e195749/Desktop/三年前期/機械学習/winequality-white.csv", sep=";")
ds.info()

#ヒストグラムで特徴量のばらつきを確認
ds.hist()
plt.tight_layout()
plt.show()

#データセットをnumpy行列に変換
ds_np = ds.to_numpy()

X = ds_np[:,:11]
y = ds_np[:, np.newaxis, 11] 

#トレーニングデータとテストデータの分類
num_of_training = 2449 
X_train = X[:num_of_training]
print(X_train.shape) 

X_test = X[num_of_training:]
print(X_test.shape) 

y_train = y[:num_of_training] 
print(y_train.shape) 

y_test = y[num_of_training:]
print(y_test.shape)

print("前処理なしの行列")
print("トレーニングデータ")
print(X_train[:1])
print("----------------------------------")

#標準化
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
X_train_std = standardScaler.fit_transform(X_train)
X_test_std = standardScaler.fit_transform(X_test)

print("標準化後の行列")
print("トレーニングデータ")
print(X_train_std[:1])
print("----------------------------------")

#正規化
from sklearn.preprocessing import Normalizer
train_normalizer = Normalizer().fit_transform(X_train)
test_normalizer = Normalizer().fit_transform(X_test)

print("正規化後の行列")
print("トレーニングデータ")
print(train_normalizer[:1])
print("----------------------------------")

#標準化->正規化
XX_train_normalizer = Normalizer().fit_transform(X_train_std)
XX_test_normalizer = Normalizer().fit_transform(X_test_std)
print("標準化->正規化後のトレーニングデータ")
print(XX_train_normalizer[:1])
print("----------------------------------")

#正規化->標準化
standardScaler = StandardScaler()
XXX_train_std = standardScaler.fit_transform(train_normalizer)
XXX_test_std = standardScaler.fit_transform(test_normalizer)
print("正規化->標準化後のトレーニングデータ")
print(XXX_train_std[:1])
print("----------------------------------")

#機械学習と精密度
#オリジナルの結果
clf = KNeighborsClassifier(n_neighbors=100)
clf.fit(X_train, y_train.ravel())
predicted = clf.predict(X_test)
print("前処理を行わない場合の正答率")
print(metrics.accuracy_score(y_test, predicted))

#標準化の結果
clf = KNeighborsClassifier(n_neighbors=100)
clf.fit(X_train_std, y_train.ravel())
predicted = clf.predict(X_test_std)
print("標準化を行った場合の正答率")
print(metrics.accuracy_score(y_test, predicted))

#正規化の結果
clf = KNeighborsClassifier(n_neighbors=100)
clf.fit(train_normalizer, y_train.ravel())
predicted = clf.predict(test_normalizer)
print("正規化を行った場合の正答率")
print(metrics.accuracy_score(y_test, predicted))

#標準化->正規化の結果
clf = KNeighborsClassifier(n_neighbors=100)
clf.fit(XX_train_normalizer, y_train.ravel())
predicted = clf.predict(XX_test_normalizer)
print("標準化->正規化を行った場合の正答率")
print(metrics.accuracy_score(y_test, predicted))

#正規化->標準化の結果
clf = KNeighborsClassifier(n_neighbors=100)
clf.fit(XXX_train_std, y_train.ravel())
predicted = clf.predict(XXX_test_std)
print("正規化->標準化を行った場合の正答率")
print(metrics.accuracy_score(y_test, predicted))
