import os

import keras
import numpy
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

x_train = np.asarray([[.5], [1.0], [.4], [5], [25]])
y_train = np.asarray([.25, .5, .2, 2.5, 12.5])

opt = keras.optimizers.Adam(lr=0.01)

model = Sequential()
model.add(Dense(1, activation="relu", input_shape=(x_train.shape[1:])))
model.add(Dense(9, activation="relu"))
model.add(Dense(1, activation="relu"))

# model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
# model.fit(x_train, y_train, shuffle=True, epochs=10)
dataset = {
    'x': x_train,
    'y': y_train,
    'x_val': x_train,
    'y_val': y_train
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=opt,loss='mean_squared_error',dataset=dataset,iters=10,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', x_train)
numpy.save(Y_test_name+'.npy', y_train)
print(model.predict(np.asarray([[5]])))
