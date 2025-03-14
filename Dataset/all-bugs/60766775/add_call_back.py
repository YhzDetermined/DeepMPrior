import os

import numpy
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import optimizers

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

x_train, y_train = make_regression(1000, 8, n_targets=4)

scaler = StandardScaler()
input_shape = x_train[0].shape
x_train_std = scaler.fit_transform(x_train)

model = Sequential()
model.add(layers.Dense(32, activation='sigmoid', input_shape=input_shape))
model.add(Dropout(0.1))
model.add(layers.Dense(20, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(layers.Dense(15, activation='sigmoid'))
model.add(Dropout(0.1))

model.add(layers.Dense(4, activation='softmax'))
sgd = optimizers.SGD(learning_rate=0.01, momentum=0.87, nesterov=True)
# model.compile(loss='mean_squared_error',
#               optimizer=sgd)
# history = model.fit(x_train_std, y_train, validation_split=0.1, epochs=100, batch_size=1)
X_train_std, X_val, y_train, y_val = train_test_split(x_train_std, y_train, test_size=0.1, random_state=42)
dataset = {
    'x': X_train_std,
    'y': y_train,
    'x_val': X_val,
    'y_val': y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=sgd,loss='mean_squared_error',dataset=dataset,iters=100,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_val)
numpy.save(Y_test_name+'.npy', y_val)