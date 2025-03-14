import os

import numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

x, y = make_classification(1000, 34)

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=34))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='relu'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x, y, epochs=2, batch_size=200)
dataset = {
    'x': x,
    'y': y,
    'x_val': x,
    'y_val': y
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='rmsprop',loss='binary_crossentropy',dataset=dataset,iters=2,batch_size=200,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', x)
numpy.save(Y_test_name+'.npy', y)