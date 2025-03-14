import os

import numpy
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_classification(1000, 200 * 200 * 3, n_classes=3, n_informative=3)
X = X.reshape(-1, 200, 200, 3)
y = to_categorical(y)

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3), activation='relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=3, activation='sigmoid'))
# Compiling the CNN
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# classifier.fit(X, y, epochs=5, validation_split=0.25)
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.25, random_state=42)
dataset = {
    'x': X_train,
    'y': Y_train,
    'x_val': X_val,
    'y_val': Y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=classifier,optimizer='adam',loss='categorical_crossentropy',dataset=dataset,iters=5,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_val)
numpy.save(Y_test_name+'.npy', Y_val)