import os

from keras import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train
import numpy
X, y = make_classification(1000, 10, n_classes=5, n_informative=5)
X = X.reshape(-1, 10, 1)
train_data, validation_data, train_labels, validation_labels = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_labels = to_categorical(train_labels, 5)
validation_labels = to_categorical(validation_labels, 5)

#model.fit(train_data, train_labels, epochs=50, validation_data=(validation_data, validation_labels))
dataset = {
    'x': train_data,
    'y': train_labels,
    'x_val': validation_data,
    'y_val': validation_labels
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer='rmsprop', loss='categorical_crossentropy', dataset=dataset, iters=50, batch_size=32,
               callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)