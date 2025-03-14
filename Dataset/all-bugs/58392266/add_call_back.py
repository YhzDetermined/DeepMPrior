import os

import numpy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_classification(100, 57 * 57 * 3)
X = X.reshape(-1, 57, 57, 3)

vgg16_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(57, 57, 3)))
vgg16_model.summary()

model = Sequential()

for layer in vgg16_model.layers:
    layer.trainable = False
    model.add(layer)

model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='softmax'))

# model.compile(loss='binary_crossentropy',
#               optimizer=Adam(learning_rate=1e-5),
#               metrics=['accuracy'])
#
# model.fit(X, y, epochs=5, validation_split=0.1, verbose=2)
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.1, random_state=42)
dataset = {
    'x': X_train,
    'y': Y_train,
    'x_val': X_val,
    'y_val': Y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=Adam(learning_rate=1e-5),loss='binary_crossentropy',dataset=dataset,iters=5,batch_size=32,callbacks=[],verb=2,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)