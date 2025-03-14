import os

import numpy
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
from sklearn.datasets import make_regression

from feature_extraction.Config import train_model_name, X_test_name, Y_test_name, params
from feature_extraction.Util.Utils import my_model_train

X, y = make_regression(1000, 150 * 150 * 1, n_targets=4, random_state=42)
X = X.reshape(-1, 150, 150, 1)

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(150, 150, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(4))

# model.compile(loss="mean_squared_error", optimizer='adam', metrics=[])
#
# model.fit(X, y, batch_size=1, validation_split=0, epochs=30, verbose=1)
dataset = {
    'x': X,
    'y': y,
    'x_val': X,
    'y_val': y
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='binary_crossentropy',dataset=dataset,iters=30,batch_size=1,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)
