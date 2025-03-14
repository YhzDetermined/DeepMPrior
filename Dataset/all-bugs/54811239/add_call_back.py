import os

import numpy
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X = [[[8, 0, 18, 10],
      [9, 0, 20, 7],
      [7, 0, 17, 12]],
     [[7, 0, 31, 8],
      [5, 0, 22, 9],
      [7, 0, 17, 12]]]
y = [[[10],
      [7],
      [12]],
     [[8],
      [9],
      [12]]]

X = np.asarray(X)
y = np.asarray(y)

model = Sequential()
model.add(LSTM(10, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dense(50, activation='tanh'))
model.add(Dense(1, activation='tanh'))

# model.compile(optimizer='adam', loss='mse', metrics=['mse'])
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
res,model=my_model_train(model=model,optimizer='adam',loss='mse',dataset=dataset,iters=5,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)