import os

import numpy
from keras import Sequential
from keras.layers import Conv3D, BatchNormalization, LeakyReLU, MaxPooling3D, Dropout, Flatten, Dense, Activation
from sklearn.datasets import make_classification

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train
import tensorflow as tf
with tf.device('/CPU:0'):
    X, y = make_classification(50, 50 * 50 * 50)
    X = X.reshape(-1, 50, 50, 50, 1)
    shape = X.shape[1:]
    model = Sequential()
    model.add(Conv3D(64, kernel_size=(5, 5, 5), activation='linear',
                                  kernel_initializer='glorot_uniform', input_shape=shape))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(LeakyReLU(.1))
    model.add(Dropout(.25))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='linear',
                                  kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(LeakyReLU(.1))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(.25))

    model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='linear',
                                  kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(LeakyReLU(.1))
    model.add(Dropout(.25))
    model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='linear',
                                  kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(LeakyReLU(.1))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(LeakyReLU(.1))
    model.add(Dropout(.5))
    model.add(Dense(512))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(LeakyReLU(.1))
    model.add(Dropout(.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # model.compile(optimizer='adam',
    #                             loss='binary_crossentropy',
    #                             metrics=['accuracy'])
    #
    # model.fit(X, y, epochs=50, batch_size=50)
    dataset = {
    'x': X,
    'y': y,
    'x_val': X,
    'y_val': y
    }
    check_interval='epoch_1'
    save_dir = os.path.join("result_dir")
    log_dir = os.path.join("log_dir")
    res,model=my_model_train(model=model,optimizer='adam',loss='binary_crossentropy',dataset=dataset,iters=50,batch_size=50,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
    model.save(train_model_name+".h5")
    numpy.save(X_test_name+'.npy', X)
    numpy.save(Y_test_name+'.npy', y)