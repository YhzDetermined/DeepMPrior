import os

import numpy
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from feature_extraction.Config import train_model_name, X_test_name, Y_test_name, params
from feature_extraction.Util.Utils import my_model_train

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X = []
y = []
for i in range(len(y_train)):
    if y_train[i] < 3:
        X.append(X_train[i])
        y.append(y_train[i])
for i in range(len(y_test)):
    if y_test[i] < 3:
        X.append(X_test[i])
        y.append(y_test[i])
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train = np.expand_dims(X_train, axis=3)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('softmax'))

# model.compile(loss="binary_crossentropy",
#               optimizer="adam",
#               metrics=['accuracy'])
#
# model.fit(X_train, y_train, batch_size=24, epochs=3, validation_split=0.1)
X_train, X_val, y_train, Y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_val,
    'y_val': Y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer="adam",loss='binary_crossentropy',dataset=dataset,iters=3,batch_size=24,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)
predictions = model.predict(X_test)

