import os

from keras.models import Sequential
from keras.layers import Dense
import numpy

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

numpy.random.seed(7)

# load and read dataset
dataset = numpy.genfromtxt('dataset.csv', delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 2:4]
Y = dataset[:, 1]
print("Variables: \n", X)
print("Target_outputs: \n", Y)
# create model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()
# Compile model
# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['MSE'])
# Fit the model
# model.fit(X, Y, epochs=500, batch_size=10)
# make predictions (test)
dataset = {
    'x': X,
    'y': Y,
    'x_val': X,
    'y_val': Y
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer='sgd', loss='mean_squared_error', dataset=dataset, iters=500, batch_size=10,
               callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', Y)
F = model.predict(X)
print("Predicted values: \n", F)