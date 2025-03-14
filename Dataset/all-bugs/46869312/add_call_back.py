# Keras imports
import numpy
from keras.models import Sequential
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.initializers import RandomUniform
from keras.optimizers import Adam

# System imports
import os
import numpy as np

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_model(X, y, num_streams, num_stages):
    '''
    STEP1: Initialize the Model
    '''

    tr_X, ts_X, tr_y, ts_y = train_test_split(X, y, train_size=.8)
    model = initialize_model(num_streams, num_stages)

    '''
    STEP2: Train the Model
    '''
    # model.compile(loss='binary_crossentropy',
    #               optimizer=Adam(lr=1e-3),
    #               metrics=['accuracy'])
    # model.fit(tr_X, tr_y,
    #           validation_data=(ts_X, ts_y),
    #           epochs=3,
    #           batch_size=200)
    dataset = {
        'x': tr_X,
        'y': tr_y,
        'x_val': ts_X,
        'y_val': ts_y
    }
    check_interval = 'epoch_1'
    save_dir = os.path.join("result_dir")
    log_dir = os.path.join("log_dir")
    res, model = my_model_train(model=model, optimizer=Adam(lr=1e-3), loss='binary_crossentropy', dataset=dataset, iters=3,
                                batch_size=200,
                                callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
                                checktype=check_interval, log_dir=log_dir)
    model.save(train_model_name + ".h5")
    numpy.save(X_test_name + '.npy', ts_X)
    numpy.save(Y_test_name + '.npy', ts_y)

def initialize_model(num_streams, num_stages):
    model = Sequential()
    hidden_units = 2 ** (num_streams + 1)
    # init = VarianceScaling(scale=5.0, mode='fan_in', distribution='normal')
    init_bound1 = np.sqrt(3.5 / ((num_stages + 1) + num_stages))
    init_bound2 = np.sqrt(3.5 / ((num_stages + 1) + hidden_units))
    init_bound3 = np.sqrt(3.5 / (hidden_units + 1))
    # drop_out = np.random.uniform(0, 1, 3)

    # This is the input layer (that's why you have to state input_dim value)
    model.add(Dense(num_stages,
                    input_dim=num_stages,
                    activation='relu',
                    kernel_initializer=RandomUniform(minval=-init_bound1, maxval=init_bound1)))

    model.add(Dense(hidden_units,
                    activation='relu',
                    kernel_initializer=RandomUniform(minval=-init_bound2, maxval=init_bound2)))

    # model.add(Dropout(drop_out[1]))

    # This is the output layer
    model.add(Dense(1,
                    activation='sigmoid',
                    kernel_initializer=RandomUniform(minval=-init_bound3, maxval=init_bound3)))

    return model


X, y = make_regression(125000, 64)
mx = max(y)
mn = min(y)
y = 2 * ((y - mn) / (mx - mn)) - 1
train_model(X, y, 4, 64)
