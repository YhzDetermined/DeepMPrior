import os

import numpy
from keras.utils import to_categorical
from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X_train, y_train = make_classification(1000, n_features=33, n_classes=122, n_informative=20)
y_train = to_categorical(y_train)

model = Sequential()
model.add(Dense(50, input_dim=33, kernel_initializer='uniform', activation='relu'))
for u in range(3):  # how to efficiently add more layers
    model.add(Dense(33, kernel_initializer='uniform', activation='relu'))
model.add(Dense(122, kernel_initializer='uniform', activation='sigmoid'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# This line of code is an update to the question and may be responsible
# model.fit(X_train, y_train, epochs=35, batch_size=20, validation_split=0.2)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_val,
    'y_val': y_val
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer='adam', loss='categorical_crossentropy', dataset=dataset, iters=35, batch_size=20,
               callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_val)
numpy.save(Y_test_name+'.npy', y_val)