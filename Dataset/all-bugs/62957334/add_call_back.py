import os

import numpy
from keras import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPool3D
from keras.optimizers import Nadam
from sklearn.datasets import make_classification

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_classification(100, 100 * 100 * 3 * 6)
X = X.reshape(-1, 100, 100, 3, 6)

model = Sequential()
model.add(Conv3D(32, (3, 3, 3), input_shape=(100, 100, 3, 6), activation='linear', padding='same'))
model.add(MaxPool3D((2, 2, 2), padding='same'))
model.add(Conv3D(32, (3, 3, 3), activation='linear', padding='same'))
model.add(MaxPool3D((2, 2, 2), padding='same'))
model.add(Conv3D(32, (3, 3, 3), activation='linear', padding='same'))
model.add(Flatten())
model.add(Dense(10, activation='linear'))
model.add(Dense(units=1, activation='sigmoid'))

# model.compile(optimizer=Nadam(), loss='mape')
# model.fit(X, y, epochs=10, shuffle=True, batch_size=20)
dataset = {
    'x': X,
    'y': y,
    'x_val': X,
    'y_val': y
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=Nadam(),loss='mape',dataset=dataset,iters=10,batch_size=20,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)