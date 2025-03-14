import os

import  keras
import numpy
import  tensorflow as tf
from keras import  layers
from keras.layers import Input, Dense
from keras.models import Model,Sequential
import numpy as np
from  keras.optimizers import  Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X,y=make_classification(1000,19900)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

myModel = keras.Sequential([
    keras.layers.Dense(1000,activation=tf.nn.relu,input_shape=(19900,)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.softmax)
])

# myModel.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# myModel.fit(X_train, y_train, epochs=100,batch_size=1000)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_test,
    'y_val': y_test
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=myModel,optimizer='adam',loss='categorical_crossentropy',dataset=dataset,iters=100,batch_size=1000,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)
test_loss,test_acc=myModel.evaluate(X_test,y_test)