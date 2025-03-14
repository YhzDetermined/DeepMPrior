import os

import numpy
from keras import Sequential
from keras.layers.core import Activation
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.datasets import make_classification
import numpy as np

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_classification(1000, 900)
X = np.reshape(X, (-1, 30, 30, 1))

model = Sequential()
model.add(Conv2D(96, kernel_size=11, padding="same", input_shape=(30, 30, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=3, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=3, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("softmax"))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(X, y, epochs=5)
dataset = {
    'x': X,
    'y': y,
    'x_val': X,
    'y_val': y
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='binary_crossentropy',dataset=dataset,iters=5,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)
