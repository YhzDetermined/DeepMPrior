import os

import numpy
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

data = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
output = [[1, 0], [0, 1], [0, 1], [1,0]]

model = Sequential()
model.add(LSTM(10, input_shape=(1, 2), return_sequences=True))
model.add(LSTM(10))
model.add(Dense(2))

# model.compile(loss='mae', optimizer='adam')
# model.fit(np.asarray(data), np.asarray(output), epochs=50)
dataset = {
    'x': data,
    'y': output,
    'x_val': data,
    'y_val': output
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='mae',dataset=dataset,iters=50,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', data)
numpy.save(Y_test_name+'.npy', output)
# print(model.predict_classes(np.asarray(data)))
