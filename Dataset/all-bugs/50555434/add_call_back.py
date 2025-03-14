import os

import numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, Y = make_regression(1000, 28, n_targets=6)
for row_index in range(0, len(Y)):
    Y[row_index] = Y[row_index] / float(sum(Y[row_index]))

# create model
model = Sequential()
model.add(Dense(20, input_dim=28, kernel_initializer='normal', activation='relu'))
model.add(Dense(15, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='sigmoid'))

# Compile model
# model.compile(optimizer="adam", loss='mae')
# Fit the model
# model.fit(X, Y, epochs=2000, verbose=2, validation_split=0.15)

# calculate predictions
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
dataset = {
    'x': X_train,
    'y': Y_train,
    'x_val': X_val,
    'y_val': Y_val
}
check_interval='epoch_4'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer="adam",loss='mae',dataset=dataset,iters=2000,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', Y)
predictions = model.predict(X)
