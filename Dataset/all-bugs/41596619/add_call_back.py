import os

import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense
import numpy

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# split into input (X) and output (Y) variables
X = numpy.array([[0.5, 1, 1], [0.9, 1, 2], [0.8, 0, 1], [0.3, 1, 1], [0.6, 1, 2], [0.4, 0, 1], [0.9, 1, 7], [0.5, 1, 4],
                 [0.1, 0, 1], [0.6, 1, 0], [1, 0, 0]])
y = numpy.array([[1], [1], [1], [2], [2], [2], [3], [3], [3], [0], [0]])

# create model
model = Sequential()
model.add(Dense(3, input_dim=3, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
opt = keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Fit the model
# model.fit(X, y, epochs=150)
dataset = {
    'x': X,
    'y': y,
    'x_val': X,
    'y_val': y
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer=opt, loss='binary_crossentropy', dataset=dataset, iters=150, batch_size=32,
               callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)
# calculate predictions
predictions = model.predict(X)
# round predictions
predictions_class = predictions.argmax(axis=-1)
print(predictions_class)
