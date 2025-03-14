import os

import numpy
import numpy as np
from keras import models
from keras import layers
import tensorflow as tf

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

np.random.seed(123)
dataSize = 100000
xdata = np.zeros((dataSize, 30))
ydata = np.zeros(dataSize)
for i in range(dataSize):
    vec = (np.random.rand(30) * 2) - 1
    xdata[i] = vec
    ydata[i] = vec[1]
model = models.Sequential()
model.add(layers.Dense(1, activation="relu", input_shape=(30, )))
model.add(layers.Dense(1, activation="sigmoid"))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
lossObject = tf.keras.losses.MeanSquaredError()
# model.compile(optimizer=optimizer, loss=lossObject)
# model.fit(xdata, ydata, epochs=200, batch_size=32)
dataset = {
    'x': xdata,
    'y': ydata,
    'x_val': xdata,
    'y_val': ydata
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=optimizer,loss=lossObject,dataset=dataset,iters=200,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', xdata)
numpy.save(Y_test_name+'.npy', ydata)