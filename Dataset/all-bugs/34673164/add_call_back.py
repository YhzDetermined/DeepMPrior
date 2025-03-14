import os
import numpy
from keras.utils import to_categorical
from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X_train, y_train = make_classification(1000, 14)
y_train = to_categorical(y_train)

model = Sequential()
model.add(Dense(64, input_dim=14, kernel_initializer='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, kernel_initializer='uniform'))
model.add(Activation('softmax'))

sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer=sgd)
# model.fit(X_train, y_train, epochs=20, batch_size=16)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_train,
    'y_val': y_train
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer=sgd, loss='mean_squared_error', dataset=dataset, iters=20, batch_size=16,
               callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_train)

numpy.save(Y_test_name+'.npy', y_train)
