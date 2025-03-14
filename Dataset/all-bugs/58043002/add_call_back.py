import os

import numpy
import numpy as np
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Masking, LSTM, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_classification(1000, 70)
X = np.reshape(X, (1000, 10, 7))
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

model = Sequential()
#CNN
model.add(Conv1D(filters=64, kernel_size=3, activation='sigmoid', input_shape=(None, 7)))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(filters=64, kernel_size=2, activation='sigmoid'))

#RNN
model.add(Masking(mask_value=0.0))
model.add(LSTM(8))
model.add(Dense(2, activation='softmax'))

opt = Adam(lr=0.0001)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
#
# print(model.fit(X_train, y_train, epochs=100, verbose=2, batch_size=50))

dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_test,
    'y_val': y_test
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=opt,loss='categorical_crossentropy',dataset=dataset,iters=100,batch_size=50,callbacks=[],verb=2,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)
score, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(score, accuracy)