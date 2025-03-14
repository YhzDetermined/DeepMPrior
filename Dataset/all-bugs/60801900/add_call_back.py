import os

import numpy
from keras import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

number_of_samples = 100
MAX_CONTEXTS = 430
X, y = make_classification(number_of_samples, MAX_CONTEXTS * 3)
scaler = MinMaxScaler(feature_range=(-1.9236537371711413, 1.9242677998256246))
X = scaler.fit_transform(X, y)
X = X.reshape(-1, MAX_CONTEXTS, 3)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(MAX_CONTEXTS, 3)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='relu'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

X_train, X_val, Y_train, Y_val = train_test_split(X, X, test_size=0.2, random_state=42)
history = model.fit(X, X, epochs=15, batch_size=2, verbose=1, shuffle=True, validation_split=0.2)
dataset = {
    'x': X_train,
    'y': Y_train,
    'x_val': X_val,
    'y_val': Y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='mse',dataset=dataset,iters=15,batch_size=2,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_val)
numpy.save(Y_test_name+'.npy', Y_val)
