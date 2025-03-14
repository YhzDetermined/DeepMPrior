import os

from keras import Input, regularizers, Model
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.datasets import make_classification
import numpy
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

input_shape = (128, 128, 3)
X, y = make_classification(500, 128 * 128 * 3)
X = X.reshape(-1, 128, 128, 3)

inp = Input(shape=input_shape)
out = Conv2D(16, (5, 5), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(inp)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.5)(out)

out = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.5)(out)

out = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = Dropout(0.5)(out)

out = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)

out = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)

out = Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Conv2D(512, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(0.01), padding='same')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)

out = Flatten()(out)
out = Dropout(0.5)(out)
dense1 = Dense(1, activation="softmax")(out)
model = Model(inputs=inp, outputs=dense1)

# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])
# model.fit(X, y, epochs=10, validation_split=0.1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
dataset = {
    'x': X_train,
    'y': y_train,
    'x_val': X_val,
    'y_val': y_val
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer=Adam(), loss='binary_crossentropy', dataset=dataset, iters=10, batch_size=32,
               callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_val)
numpy.save(Y_test_name+'.npy', y_val)