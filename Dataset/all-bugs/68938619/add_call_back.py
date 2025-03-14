import os

import numpy
from keras.utils import normalize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import keras

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

features = 10
X, y = make_classification(1000, 10)
X = normalize(X)
X = X.reshape(-1, 10, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

Model = keras.Sequential([
    keras.layers.LSTM(80, input_shape=(features, X_train.shape[2]),
                      activation='sigmoid', recurrent_activation='hard_sigmoid'),
    keras.layers.Dense(1, activation="softmax")
])

#Model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

# Training the model

#Model.fit(X_train, y_train, epochs=10, batch_size=32)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_test,
    'y_val': y_test
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=Model,optimizer='rmsprop',loss='mse',dataset=dataset,iters=10,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)
Model.summary()

# Final evaluation of the model
scores = Model.evaluate(X_test, y_test, verbose=0)
print('/n')
print("Accuracy: %.2f%%" % (scores[1] * 100))
