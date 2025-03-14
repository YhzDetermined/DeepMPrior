import os

import numpy
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

dataset = loadtxt("dataset.csv", delimiter=",")
X = dataset[:, 0:2]
y = dataset[:, 2]
model = Sequential()
model.add(Dense(196, input_dim=2, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X, y, epochs=600, batch_size=10)
dataset = {
    'x': X,
    'y': y,
    'x_val': X,
    'y_val': y
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='binary_crossentropy',dataset=dataset,iters=600,batch_size=10,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))
