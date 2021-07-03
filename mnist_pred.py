'''3.6 手書き文字認識'''
%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/15_Zero_Making_Deeplearning
import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from PIL import Image


#%% データのダウンロード
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

#%% 画像の表示
def img_show(img):
    # pil用のデータオブジェクトに変換
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = x_train[0]
print(label)

print(img.shape)
# 画像を表示するためにreshape
img = img.reshape(28, 28)
print(img.shape)

img_show(img)


#%% モデルの定義
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = (x @ W1) + b1
    z1 = sigmoid(a1)
    a2 = (z1 @ W2) + b2
    z2 = sigmoid(a2)
    a3 = (z2 @ W3) + b3
    y = softmax(a3)
    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1

print('Accuracy:' + str(float(accuracy_cnt) /len(x)))
