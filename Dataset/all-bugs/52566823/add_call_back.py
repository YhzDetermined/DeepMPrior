import os

import numpy
from keras import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_regression(1000, 5, random_state=75)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=75)

model = Sequential()
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.1)
# model.compile(sgd, 'mse')
# model.fit(X_train, y_train, 32, 100, shuffle=False)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_test,
    'y_val': y_test
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=sgd,loss='mse',dataset=dataset,iters=100,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)