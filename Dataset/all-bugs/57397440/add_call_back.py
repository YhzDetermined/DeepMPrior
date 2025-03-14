import os

import numpy
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
import keras.activations as tfnn

from sklearn.datasets import load_breast_cancer

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

data, target = load_breast_cancer(return_X_y=True)

model = Sequential()

model.add(Dense(30, activation=tfnn.relu, input_dim=30))
model.add(BatchNormalization(axis=1))

model.add(Dense(60, activation=tfnn.relu))
model.add(BatchNormalization(axis=1))

model.add(Dense(1, activation=tfnn.softmax))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(data, target, epochs=6)
dataset = {
    'x': data,
    'y': target,
    'x_val': data,
    'y_val': target
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='binary_crossentropy',dataset=dataset,iters=6,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', data)
numpy.save(Y_test_name+'.npy', target)

