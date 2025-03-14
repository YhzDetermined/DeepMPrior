import os

import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
X, y = make_classification(100, 230 * 230 * 3, random_state=42)
X = X.reshape(-1, 230, 230, 3)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Conv2D(32, 1, 1, input_shape=(230, 230, 3)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(1))

# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_val,
    'y_val': y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='rmsprop',loss='binary_crossentropy',dataset=dataset,iters=50,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_val)
numpy.save(Y_test_name+'.npy', y_val)