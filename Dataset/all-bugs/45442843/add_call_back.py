import os

import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_regression(100, 1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Activation('sigmoid'))
model.add(Dense(2, kernel_initializer='normal', activation='softmax'))

# model.compile(loss='mean_absolute_error', optimizer='rmsprop')
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_test,
    'y_val': y_test
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer='rmsprop', loss='mean_absolute_error', dataset=dataset, iters=10, batch_size=200,
               callbacks=[], verb=2, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)