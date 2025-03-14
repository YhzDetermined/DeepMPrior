import os

import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

dataset = np.loadtxt("cars.csv", delimiter=",")
x = dataset[:, 0:5]
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = dataset[:, 5]
y = np.reshape(y, (-1, 1))

model = Sequential()
model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'accuracy'])
# model.fit(x, y, epochs=150, batch_size=50, verbose=1, validation_split=0.2)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
dataset = {
    'x': x_train,
    'y': y_train,
    'x_val': x_val,
    'y_val': y_val
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer='adam', loss='mse', dataset=dataset, iters=150, batch_size=50,
               callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy',x)
numpy.save(Y_test_name+'.npy', y)