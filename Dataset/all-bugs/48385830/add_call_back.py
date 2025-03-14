import os

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers.initializers_v1 import RandomNormal
from keras.utils import to_categorical

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

# import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# input image dimensions
img_rows, img_cols = 28, 28

x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
input_shape = (img_rows * img_cols,)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)

# Construct model
# 784 * 30 * 10
# Normal distribution for weights/biases
# Stochastic Gradient Descent optimizer
# Mean squared error loss (cost function)
model = Sequential()
layer1 = Dense(30,
               input_shape=input_shape,
               kernel_initializer=RandomNormal(stddev=1),
               bias_initializer=RandomNormal(stddev=1))
model.add(layer1)
layer2 = Dense(10,
               kernel_initializer=RandomNormal(stddev=1),
               bias_initializer=RandomNormal(stddev=1))
model.add(layer2)
print('Layer 1 input shape: ', layer1.input_shape)
print('Layer 1 output shape: ', layer1.output_shape)
print('Layer 2 input shape: ', layer2.input_shape)
print('Layer 2 output shape: ', layer2.output_shape)

model.summary()
# model.compile(optimizer=SGD(learning_rate=3.0),
#               loss='mean_squared_error',
#               metrics=['accuracy'])
#
# # Train
# model.fit(x_train,
#           y_train,
#           batch_size=10,
#           epochs=30,
#           verbose=2)
dataset = {
    'x': x_train,
    'y': y_train,
    'x_val': x_test,
    'y_val': y_test
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer=SGD(learning_rate=3.0), loss='mean_squared_error', dataset=dataset, iters=30, batch_size=10,
               callbacks=[], verb=2, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', x_test)
numpy.save(Y_test_name+'.npy', y_test)
# Run on test data and output results
result = model.evaluate(x_test,
                        y_test,
                        verbose=1)
print('Test loss: ', result[0])
print('Test accuracy: ', result[1])
