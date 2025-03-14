import os

import keras
import numpy
import numpy as np
from keras.utils import to_categorical
from sklearn.datasets import make_classification

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X_train, y_train = make_classification(100, 6000, n_classes=20, n_informative=10)
X_train = np.reshape(X_train, (100, 20, 300))
y_train = to_categorical(y_train)

nnmodel = keras.Sequential()
nnmodel.add(keras.layers.InputLayer(input_shape=(20, 300)))
nnmodel.add(keras.layers.Dense(units=300, activation="relu"))
nnmodel.add(keras.layers.Dense(units=20, activation="relu"))
nnmodel.add(keras.layers.Dense(units=1, activation="sigmoid"))

#nnmodel.compile(optimizer='adam',
#                loss='CategoricalCrossentropy',
#                metrics=['accuracy'])
#nnmodel.fit(X_train, y_train, epochs=10, batch_size=1)

dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_train,
    'y_val': y_train
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=nnmodel,optimizer='adam',loss='CategoricalCrossentropy',dataset=dataset,iters=10,batch_size=1,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_train)
numpy.save(Y_test_name+'.npy', y_train)
for layer in nnmodel.layers:
    print(layer.output_shape)
