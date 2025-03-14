import os

import numpy
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_classification(1000, 22)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(48, input_shape=(22,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))
optim = Adam(learning_rate=0.0001)
# model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=20, batch_size=5, validation_data=(X_test, y_test))
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_test,
    'y_val': y_test
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=optim,loss='binary_crossentropy',dataset=dataset,iters=20,batch_size=5,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)