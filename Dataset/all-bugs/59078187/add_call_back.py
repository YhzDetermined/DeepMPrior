import os

import numpy
import numpy as np
import keras
from keras import Sequential,layers,models
import numpy as np

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

# Load the dataset

x_train = np.linspace(0,10,1000).reshape(-1, 1)
y_train = np.power(x_train,2.0)

x_test = np.linspace(8,12,100).reshape(-1, 1)
y_test = np.power(x_test,2.0)

"""Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:"""

model = keras.models.Sequential([
    keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(10, activation='softmax')
    ])

# model.compile(optimizer='adam',
#                             loss='mse',
#                             metrics=['mae'])

"""Train and evaluate the model:"""
#
# model.fit(x_train, y_train, epochs=5)
dataset = {
    'x': x_train,
    'y': y_train,
    'x_val': x_test,
    'y_val': y_test
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='mse',dataset=dataset,iters=5,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', x_test)
numpy.save(Y_test_name+'.npy', y_test)
model.evaluate(x_test,  y_test, verbose=2)
