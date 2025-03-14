import os

import numpy
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from sklearn.datasets import make_classification

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X_train, y_train = make_classification(1000, 7)

model = Sequential()
model.add(Dense(16, input_shape=(7,)))
model.add(Activation('relu'))
model.add(Activation('relu'))
model.add(Dense(1))  # 2 outputs possible (ok or nok)
model.add(Activation('relu'))
# model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=5)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_train,
    'y_val': y_train
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer='sgd', loss='mse', dataset=dataset, iters=5, batch_size=32,
               callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_train)
numpy.save(Y_test_name+'.npy', y_train)