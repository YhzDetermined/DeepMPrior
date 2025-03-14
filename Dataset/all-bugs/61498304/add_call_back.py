import os

import numpy
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_classification(1000, 32 * 32 * 3)
X = X.reshape(-1, 32, 32, 3)
y = to_categorical(y)

model = Sequential()
# Layer 1
# Conv Layer 1
model.add(Conv2D(filters=6,
                 kernel_size=5,
                 strides=1,
                 activation='relu',
                 input_shape=(32, 32, 3)))
# Pooling layer 1
model.add(MaxPooling2D(pool_size=2, strides=2))
# Layer 2
# Conv Layer 2
model.add(Conv2D(filters=16,
                 kernel_size=5,
                 strides=1,
                 activation='relu',
                 input_shape=(14, 14, 6)))
# Pooling Layer 2
model.add(MaxPooling2D(pool_size=2, strides=2))
# Flatten
model.add(Flatten())
# Layer 3
# Fully connected layer 1
model.add(Dense(units=128, activation='relu', kernel_initializer='uniform',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(rate=0.2))
# Layer 4
# Fully connected layer 2
model.add(Dense(units=64, activation='relu', kernel_initializer='uniform',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(rate=0.2))

# layer 5
# Fully connected layer 3
model.add(Dense(units=64, activation='relu', kernel_initializer='uniform',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(rate=0.2))

# layer 6
# Fully connected layer 4
model.add(Dense(units=64, activation='relu', kernel_initializer='uniform',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(rate=0.2))

# Layer 7
# Output Layer
model.add(Dense(units=2, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(X, y, epochs=25, validation_split=0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=42)
dataset = {
    'x': X_train,
    'y': Y_train,
    'x_val': X_val,
    'y_val': Y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='categorical_crossentropy',dataset=dataset,iters=25,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)
