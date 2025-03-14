import os

import numpy
import numpy as np
from keras.layers import Dense
from keras.callbacks import History
from keras import optimizers, Sequential

from feature_extraction.Config import params, train_model_name, Y_test_name, X_test_name
from feature_extraction.Util.Utils import my_model_train

X = np.array([[1], [2]])
Y = np.array([[1], [0]])

history = History()

inputDim = len(X[0])
print('input dim', inputDim)
model = Sequential()

model.add(Dense(1, activation='sigmoid', input_dim=inputDim))
model.add(Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.009, decay=1e-10, momentum=0.9, nesterov=True)
# model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.fit(X, Y, validation_split=0.1, verbose=2, callbacks=[history], epochs=20, batch_size=32)
dataset = {
    'x': X,
    'y': Y,
    'x_val': X,
    'y_val': Y
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer=sgd, loss='binary_crossentropy', dataset=dataset, iters=20, batch_size=32,
               callbacks=[history], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', Y)
