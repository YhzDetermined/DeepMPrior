import os

import numpy
import numpy as np
from keras import Sequential
from keras.layers import GRU, Dense

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

data = [[[0], [0], [0], [1], [1], [0], [1], [1]]]
output = [[[0], [0], [0], [1], [0], [1], [1], [0]]]

model = Sequential()
model.add(GRU(10, input_shape=(8, 1), return_sequences=True))
model.add(GRU(10, return_sequences=True))
model.add(Dense(1, activation='softmax'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(np.asarray(data), np.asarray(output), epochs=3000)
dataset = {
    'x': data,
    'y': output,
    'x_val': data,
    'y_val': output
}
check_interval='epoch_6'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='binary_crossentropy',dataset=dataset,iters=3000,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', data)
numpy.save(Y_test_name+'.npy', output)