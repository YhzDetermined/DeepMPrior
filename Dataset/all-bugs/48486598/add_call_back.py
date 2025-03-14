import os

import numpy
from keras import Sequential
from keras.layers import Dense
import numpy as np

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

# declare and init arrays for training-data

X = np.arange(0.0, 10.0, 0.05)
Y = np.empty(shape=0, dtype=float)

# Calculate Y-Values
for x in X:
    Y = np.append(Y, float(0.05*(15.72807*x - 7.273893*x**2 + 1.4912*x**3 - 0.1384615*x**4 + 0.00474359*x**5)))

# model architecture
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.add(Dense(5))
model.add(Dense(1, activation='linear'))

# compile model
# model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

# train model
# model.fit(X, Y, epochs=150, batch_size=10)
dataset = {
    'x': X,
    'y': Y,
    'x_val': X,
    'y_val': Y
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer='adam', loss='mean_squared_error', dataset=dataset, iters=150, batch_size=10,
               callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy',X)
numpy.save(Y_test_name+'.npy', Y)
# declare and init arrays for prediction
YPredict = np.empty(shape=0, dtype=float)

# Predict Y
YPredict = model.predict(X)

print(YPredict)
