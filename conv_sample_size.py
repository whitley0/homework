import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import os


def model_create():
    '''搭建conv神经网络'''
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32,3,1,padding='same',input_shape=(32,32,3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(32,3,1,padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(64,3,1,padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64,3,1,padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512,activation='relu'))
    model.add(keras.layers.Dense(10,activation='softmax'))
    opt = keras.optimizers.RMSprop(lr=0.002,decay=1e-5)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # 加载数据
    cifar10 = keras.datasets.cifar10
    (x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()
    x_train_all = x_train_all.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train_all = keras.utils.to_categorical(y_train_all.reshape(y_train_all.shape[0]), 10)
    y_test = keras.utils.to_categorical(y_test.reshape(y_test.shape[0]), 10)

    # 训练模型
    historys = []
    for i in range(5):
        model = model_create()
        x_train = x_train_all[:10000 * (i + 1)]
        y_train = y_train_all[:10000 * (i + 1)]
        logdir = 'cifar10_cnn_%d' % i
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        output_model_file = os.path.join(logdir, "cifar10_cnn_model_%d.h5") % i

        callbacks = [
            keras.callbacks.TensorBoard(log_dir=logdir),
            keras.callbacks.ModelCheckpoint(output_model_file, monitor='val_accuracy', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)
        ]

        history = model.fit(x_train, y_train, batch_size=512, epochs=20, validation_data=(x_test, y_test),
                            callbacks=callbacks)
        historys.append(history)

    # 画出loss曲线
    history_dict = [{}, {}, {}]
    for i in range(5):
        history_dict[0]['%d datasets' % (10000 * (i + 1))] = historys[i].history['loss']
        history_dict[1]['%d datasets' % (10000 * (i + 1))] = historys[i].history['val_loss']
        history_dict[2]['%d datasets' % (10000 * (i + 1))] = historys[i].history['accuracy']
    ylabel=['Val_loss','Loss','Accuracy']
    font = {'weight':'normal','size':20}
    for i in range(3):
        pd.DataFrame(history_dict[i]).plot(figsize=(10,10))
        plt.grid(True)
        plt.xlabel('Number of epoches',font)
        plt.ylabel(ylabel[i],font)
        if ylabel[i]=='Accuracy':
            plt.gca().set_ylim(0.95,1)
        elif ylabel[i]=='Loss':
            plt.gca().set_ylim(0,0.1)
        else:
            plt.gca().set_ylim(0,0.2)
        plt.legend(loc='best')
    plt.show()