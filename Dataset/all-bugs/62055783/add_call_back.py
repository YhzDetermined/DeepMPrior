import os

import numpy
from keras.losses import CategoricalCrossentropy
from sklearn.datasets import make_classification
from keras import models
from keras import layers
from keras import regularizers
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train
from keras.utils import to_categorical
X, y = make_classification(1000, 48 * 48, n_classes=7, n_informative=5)
X = X.reshape(-1, 48, 48, 1)
y = to_categorical(y, num_classes=7)

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3),
                        activation='relu',
                        input_shape=(48, 48, 1),
                        kernel_regularizer=regularizers.l1(0.01)))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(7))

# model.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
epochs = 100
batch_size = 64
learning_rate = 0.001
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True)
dataset = {
    'x': X_train,
    'y': Y_train,
    'x_val': X_val,
    'y_val': Y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss=CategoricalCrossentropy(from_logits=True),dataset=dataset,iters=epochs,batch_size=batch_size,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)