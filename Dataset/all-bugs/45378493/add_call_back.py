import os

import numpy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Convolution1D, BatchNormalization, LeakyReLU, Dropout, Dense, Flatten, Activation

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

X, y = make_classification(600, 10 * 4)
X = X.reshape(-1, 10, 4)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, random_state=42)

model = Sequential()
model.add(Convolution1D(input_shape=(10, 4),
                        filters=16,
                        kernel_size=4,
                        padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Convolution1D(filters=8,
                        kernel_size=4,
                        padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(1))
model.add(Activation('softmax'))

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=30, min_lr=0.000001, verbose=0)
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.__class__.__name__}")
    if isinstance(layer,Activation):
        print(f"Layer {i}: *")
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# history = model.fit(X_train, y_train,
#                     epochs=100,
#                     batch_size=128,
#                     verbose=0,
#                     validation_data=(X_test, y_test),
#                     callbacks=[reduce_lr],
#                     shuffle=True)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_test,
    'y_val': y_test
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer='adam', loss='binary_crossentropy', dataset=dataset, iters=100, batch_size=128,
               callbacks=[reduce_lr], verb=0, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', y_test)
y_pred = model.predict(X_test)
