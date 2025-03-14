import os

import numpy
from keras import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_regression(1000, 4)

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(4,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

print('Compile & fit')
# model.compile(loss='mean_squared_error', optimizer='RMSprop')
# model.fit(X, y, batch_size=128, epochs=13)
dataset = {
    'x': X,
    'y': y,
    'x_val': X,
    'y_val': y
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='RMSprop',loss='mean_squared_error',dataset=dataset,iters=13,batch_size=128,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)