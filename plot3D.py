import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


def loader_data():
    '''加载数据'''
    mnist = keras.datasets.mnist
    (x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()
    x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
    y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
    return x_train_scaled,y_train


def loader_model(file):
    '''加载模型'''
    model = keras.models.load_model(file)
    return model


def change_model_weights(file,theta1,theta2,alpha,beta):
    '''根据 theta = alpha*theta1 + beta*theta2更改神经网络权重'''
    model_test = loader_model(file)
    for i in range(1, 3):
        layer = model_test.layers[i]
        weights_test_before = layer.get_weights()[0]
        weights_test = weights_test_before + alpha * theta1[i - 1] + beta*theta2[i-1]
        model_test.layers[i].set_weights([weights_test, layer.get_weights()[1]])
    return model_test


if __name__ == '__main__':
    for i in range(6):
        file = '%d hidden_layers.h5' %(i+1)
        model = loader_model(file)
        x_train_scaled,y_train = loader_data()
        theta1 = []
        theta2 = []
        for i in range(1, 3): # 生成每一层权重对应的theta矩阵
            layer = model.layers[i]
            weights = layer.get_weights()[0]
            theta1.append(np.random.normal(0., 0.5, weights.shape))
            theta2.append(np.random.normal(0., 0.5, weights.shape))
        parameters = np.linspace(-0.5, 0.5, 51) # 生成alpha，beta
        loss_list = []
        for alpha in parameters:
            for beta in parameters:
                model_test = change_model_weights(file,theta1,theta2,alpha,beta)
                loss = model_test.evaluate(x_train_scaled, y_train)
                loss_list.append(loss)
        loss_es = []
        for loss, accuracy in loss_list: #画出3Dloss曲线
            loss_es.append(loss)
        plt.plot_surface(parameters,parameters,loss_es)
        plt.xlabel('the value of alpha')
        plt.ylabel('loss')
        plt.show()