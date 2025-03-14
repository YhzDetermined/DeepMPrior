import os

import numpy
from keras import Sequential
from keras.layers import Dense, GRU
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_regression(10000, 12)
X = X.reshape(-1, 1, 12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# creating model using Keras
model10 = Sequential()
model10.add(GRU(units=512, return_sequences=True, input_shape=(1, 12)))
model10.add(GRU(units=256, return_sequences=True))
model10.add(GRU(units=256))
model10.add(Dense(units=1, activation='sigmoid'))
# model10.compile(loss=['mse'], optimizer='adam', metrics=['mse'])
# model10.summary()

# history10 = model10.fit(X_train, y_train, batch_size=256, epochs=25, validation_split=0.20, verbose=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_val,
    'y_val': y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model10,optimizer='adam',loss=['mse'],dataset=dataset,iters=25,batch_size=256,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)
score = model10.evaluate(X_test, y_test)
print('Score: {}'.format(score))
