import os

import keras
import numpy
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_regression(1000, 424 * 424, n_targets=3, n_informative=3, random_state=42)

model = keras.Sequential()
# explicitly define SGD so that I can change the decay rate
sgd = keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.add(keras.layers.Dense(32, input_shape=(424 * 424,)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

# model.compile(loss='mean_squared_error', optimizer=sgd)
#
# model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_val,
    'y_val': y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=sgd,loss='mean_squared_error',dataset=dataset,iters=20,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)