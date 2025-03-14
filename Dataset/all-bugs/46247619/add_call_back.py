import os

from keras import Sequential
from keras.layers import Conv1D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_regression(10000, 50)
X = X.reshape(-1, 50, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = Sequential()
model.add(Conv1D(30, kernel_size=3, activation='relu', input_shape=(50, 1)))
model.add(Conv1D(40, kernel_size=3, activation='relu'))
model.add(Conv1D(120, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()
# model.compile(loss='mse', optimizer=Adam())


train_limit = 5 
batch_size = 4096
# model.fit(X_train[:train_limit], y_train[:train_limit],
#           batch_size=batch_size,
#           epochs=10**4,
#           verbose=2,
#           validation_data=(X_test[:train_limit], y_test[:train_limit]))
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_test,
    'y_val': y_test
}
check_interval = 'epoch_20'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer=Adam(), loss='mse', dataset=dataset, iters=10**4, batch_size=batch_size,
               callbacks=[], verb=2, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score)
print('Test accuracy:', score)