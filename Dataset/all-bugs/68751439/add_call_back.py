import os

import keras
import numpy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X_train, y_train = make_classification(100, 14)

# define the keras model
model1 = keras.Sequential()
model1.add(keras.layers.Dense(64, input_dim=14, activation='relu'))
model1.add(keras.layers.Dense(128, activation='relu'))
model1.add(keras.layers.Dense(64, activation='relu'))  
model1.add(keras.layers.Dense(1, activation='softmax'))

# compile the keras model
#model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
#performance1 = model1.fit(X_train, y_train, epochs=100, validation_split=0.2)
_X_train, X_val, _y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
dataset = {
    'x': _X_train,
    'y': _y_train,
    'x_val': X_val,
    'y_val': y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model1,optimizer='adam',loss='categorical_crossentropy',dataset=dataset,iters=100,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_train)
numpy.save(Y_test_name+'.npy', y_train)