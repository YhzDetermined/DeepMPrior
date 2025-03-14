import os

import numpy
import tensorflow as tf
from keras.datasets.cifar10 import load_data
from keras.utils import to_categorical

from feature_extraction.Config import train_model_name, params, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

(X_train, y_train), (X_test, y_test) = load_data()

X_train, X_test = X_train / 255., X_test / 255.
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=400, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# history = model.fit(X_train, y_train,
#                     steps_per_epoch=50,
#                     epochs=3,
#                     validation_data=(X_test, y_test),
#                     validation_steps=50)
batch_size=len(X_train) / 50
batch_size=int(batch_size)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_test,
    'y_val':y_test
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='binary_crossentropy',dataset=dataset,iters=3,batch_size=batch_size,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)