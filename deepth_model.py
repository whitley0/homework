import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# 加载数据
mnist = keras.datasets.mnist
(x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

# 数据归一化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)

#生成1，2，3，5，10，20层的全连接网络
historys = []
hidden_layers = [1,2,3,5,10,20]
for i in range(6):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28,28]))
    for j in range(hidden_layers[i]):
        model.add(keras.layers.Dense(100,activation="relu"))
    model.add(keras.layers.Dense(10,activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",optimizer='sgd',metrics=['accuracy'],
                  callbacks=[keras.callbacks.EarlyStopping(patience=5,min_delta=1e-4)])
    history = model.fit(x_train_scaled,y_train,epochs=50,validation_data=(x_valid_scaled,y_valid))
    model.save('%d hidden_layers.h5'% hidden_layers[i])
    historys.append(history)

#绘制出loss曲线
history_dict = [{},{},{}]
for i in range(4):
    history_dict[0]['%d hidden_layers' % ((i + 1)*5)] = historys[i].history['val_loss']
    history_dict[1]['%d hidden_layers' % ((i + 1)*5)] = historys[i].history['loss']
font = {'weight':'normal','size':20}
pd.DataFrame(history_dict[0]).plot(figsize=(10,10))
plt.grid(True)
plt.xlabel('Number of epoches',font)
plt.ylabel('Val_loss',font)
plt.gca().set_ylim(0,1)
plt.legend(loc='best')
pd.DataFrame(history_dict[1]).plot(figsize=(10,10))
plt.xlabel('Number of epoches',font)
plt.ylabel('Loss',font)
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.legend(loc='best')
plt.show()
