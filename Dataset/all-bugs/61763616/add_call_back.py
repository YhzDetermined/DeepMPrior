import os

import numpy
from keras.models import Sequential
from keras.layers import MaxPooling1D, Conv1D
from keras.layers import Flatten, Dense
from sklearn.datasets import make_classification

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X_train, y_train = make_classification(1000, 28 * 28)

model_05_01 = Sequential()
model_05_01.add(Conv1D(filters=16, kernel_size=12, input_shape=(X_train.shape[1], 1)))
model_05_01.add(MaxPooling1D(pool_size=4))

model_05_01.add(Conv1D(filters=32, kernel_size=12))
model_05_01.add(MaxPooling1D(pool_size=4))

model_05_01.add(Conv1D(filters=16, kernel_size=12))
model_05_01.add(MaxPooling1D(pool_size=4))

model_05_01.add(Flatten())

model_05_01.add(Dense(16, activation='relu'))
model_05_01.add(Dense(2, activation='sigmoid'))

# model_05_01.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])
#
# model_05_01.fit(X_train, y_train, epochs=5)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_train,
    'y_val': y_train
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model_05_01,optimizer='adam',loss='logcosh',dataset=dataset,iters=5,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_train)
numpy.save(Y_test_name+'.npy', y_train)
