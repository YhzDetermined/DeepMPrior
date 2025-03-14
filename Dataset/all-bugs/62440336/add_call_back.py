import os

import numpy
import tensorflow as tf
import keras
from sklearn.datasets import make_regression

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
with tf.device('/CPU:0'):
    X, y = make_regression(100, 1323000)
    X = X.reshape(-1, 1323000, 1)

    model = keras.Sequential([
        keras.layers.Conv1D(filters=100, kernel_size=10000, strides=5000, input_shape=(1323000, 1), activation='relu'),
        keras.layers.Conv1D(filters=100, kernel_size=10, strides=3, input_shape=(263, 100), activation='relu'),
        keras.layers.LSTM(1000),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dense(250, activation='relu'),
        keras.layers.Dense(1, activation='softmax')
    ])

    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               metrics=['accuracy'])

    # model.fit(X, y, epochs=1000)
    dataset = {
        'x': X,
        'y': y,
        'x_val': X,
        'y_val': y
    }
    check_interval='epoch_2'
    save_dir = os.path.join("result_dir")
    log_dir = os.path.join("log_dir")
    res,model=my_model_train(model=model,optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),dataset=dataset,iters=1000,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
                   checktype=check_interval,log_dir=log_dir)
    model.save(train_model_name+".h5")
    numpy.save(X_test_name+'.npy', X)
    numpy.save(Y_test_name+'.npy', y)