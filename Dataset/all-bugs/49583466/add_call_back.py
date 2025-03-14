import os

from keras.models import Sequential
from keras.layers import Dense
import random
import numpy
from sklearn.preprocessing import MinMaxScaler

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

regressor = Sequential()
regressor.add(Dense(units=20, activation='sigmoid', kernel_initializer='uniform', input_dim=1))
regressor.add(Dense(units=20, activation='sigmoid', kernel_initializer='uniform'))
regressor.add(Dense(units=20, activation='sigmoid', kernel_initializer='uniform'))
regressor.add(Dense(units=1))
# regressor.compile(loss='mean_squared_error', optimizer='sgd')

N = 5000
X = numpy.empty((N,))
Y = numpy.empty((N,))

for i in range(N):
    X[i] = random.uniform(-10, 10)
X = numpy.sort(X).reshape(-1, 1)

for i in range(N):
    Y[i] = numpy.sin(X[i])
Y = Y.reshape(-1, 1)

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X = X_scaler.fit_transform(X)
Y = Y_scaler.fit_transform(Y)

# regressor.fit(X, Y, epochs=2, verbose=1, batch_size=32)
dataset = {
    'x': X,
    'y': Y,
    'x_val': X,
    'y_val': Y
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=regressor, optimizer='sgd', loss='mean_squared_error', dataset=dataset, iters=2, batch_size=32,
               callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)

x = numpy.mgrid[-10:10:100*1j]
x = x.reshape(-1, 1)
y = numpy.mgrid[-10:10:100*1j]
y = y.reshape(-1, 1)
x = X_scaler.fit_transform(x)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy',x)
numpy.save(Y_test_name+'.npy', y)
for i in range(len(x)):
    y[i] = regressor.predict(numpy.array([x[i]]))
