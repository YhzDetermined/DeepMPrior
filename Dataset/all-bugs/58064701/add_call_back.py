import os

import numpy
import pandas as pd

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

dataset = pd.read_csv('./pima-indians-diabetes.csv', header=None)

X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from keras import Sequential
from keras.layers import Dense

classifier = Sequential()
# First Hidden Layer
classifier.add(Dense(units=10, activation='relu', kernel_initializer='random_normal', input_dim=8))
# Second  Hidden Layer
classifier.add(Dense(units=10, activation='relu', kernel_initializer='random_normal'))
# Output Layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='random_normal'))

# Compiling the neural network
# classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the data to the training dataset
# classifier.fit(X_train, y_train, batch_size=2, epochs=10)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_test,
    'y_val': y_test
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=classifier,optimizer='adam',loss='binary_crossentropy',dataset=dataset,iters=10,batch_size=2,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)