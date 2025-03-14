import os

import numpy
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.datasets import make_classification
import numpy as np

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_classification(100, 100)
X = np.reshape(X, (-1, 10, 10, 1))

print(X.shape)

input_shape = (10, 10, 1)
channel_dimension = 1
classes = 2

model = Sequential()
# CONV => RELU => POOL layer set
# define convolutional layers, use "ReLU" activation function
# and reduce the spatial size (width and height) with pool layers
model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape)) # 32 3x3 filters (height, width, depth)
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dimension))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) # helps prevent overfitting (25% of neurons disconnected randomly)
# (CONV => RELU) * 2 => POOL layer set (increasing number of layers as you go deeper into CNN)
model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))  # 64 3x3 filters
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dimension))
model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))  # 64 3x3 filters
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dimension))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # helps prevent overfitting (25% of neurons disconnected randomly)

# (CONV => RELU) * 3 => POOL layer set (input volume size becoming smaller and smaller)
model.add(Conv2D(128, (3, 3), padding="same", input_shape=input_shape))  # 128 3x3 filters
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dimension))
model.add(Conv2D(128, (3, 3), padding="same", input_shape=input_shape))  # 128 3x3 filters
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dimension))
model.add(Conv2D(128, (3, 3), padding="same", input_shape=input_shape))  # 128 3x3 filters
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dimension))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # helps prevent overfitting (25% of neurons disconnected randomly)

# only set of FC => RELU layers
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# sigmoid classifier (output layer)
model.add(Dense(classes))
model.add(Activation("sigmoid"))

print("[INFO] Training network...")
opt = SGD(learning_rate=0.001)
# model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
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
res,model=my_model_train(model=model,optimizer=opt,loss='sparse_categorical_crossentropy',dataset=dataset,iters=5,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)