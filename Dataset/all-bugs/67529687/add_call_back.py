import os

import numpy
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf
from feature_extraction.Config import train_model_name, X_test_name, Y_test_name, params
from feature_extraction.Util.Utils import my_model_train

with tf.device('/CPU:0'):
    X, y = make_classification(100, 224 * 224 * 3)
    X = X.reshape(-1, 224, 224, 3)
    y = to_categorical(y)

    model = Sequential()
    model.add(Conv2D(input_shape=(224, 224, 3), filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dense(units=2048, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=2, activation="sigmoid"))

    #model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    #model.fit(X, y, epochs=100, validation_split=0.1, shuffle=True)
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    dataset = {
        'x': X_train,
        'y': Y_train,
        'x_val': X_val,
        'y_val': Y_val
    }
    check_interval='epoch_1'
    save_dir = os.path.join("result_dir")
    log_dir = os.path.join("log_dir")
    res,model=my_model_train(model=model,optimizer=Adam(lr=1e-3),loss='binary_crossentropy',dataset=dataset,iters=100,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
                   checktype=check_interval,log_dir=log_dir)
    model.save(train_model_name+".h5")
    numpy.save(X_test_name+'.npy', X)
    numpy.save(Y_test_name+'.npy', y)