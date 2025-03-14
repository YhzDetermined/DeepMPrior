import os

import numpy
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

# Import the dataset
dataset = pd.read_csv("dataset.csv", header=None).values
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, 0:1], dataset[:, 1],
                                                    test_size=0.25, )
# Now we build the model
neural_network = Sequential()  # create model
neural_network.add(Dense(5, input_dim=1, activation='sigmoid'))  # hidden layer
neural_network.add(Dense(1, activation='sigmoid'))  # output layer
# neural_network.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# neural_network_fitted = neural_network.fit(X_train, Y_train, epochs=1000, verbose=0,
#                                            batch_size=X_train.shape[0], initial_epoch=0)
dataset = {
    'x': X_train,
    'y': Y_train,
    'x_val': X_test,
    'y_val': Y_test
}
check_interval = 'epoch_2'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=neural_network, optimizer='sgd', loss='mean_squared_error', dataset=dataset, iters=1000, batch_size=X_train.shape[0],
               callbacks=[], verb=0, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', Y_test)
