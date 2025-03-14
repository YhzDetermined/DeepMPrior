# import tensorflow as tf
import numpy as np
import keras
from keras import Sequential,layers,models
# Load the dataset

x_train = np.linspace(0,10,1000).reshape(-1, 1)
y_train = np.power(x_train,2.0)

x_test = np.linspace(8,12,100).reshape(-1, 1)
y_test = np.power(x_test,2.0)

"""Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:"""

model = keras.models.Sequential([
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam',
                            loss='mse',
                            metrics=['mae'])

"""Train and evaluate the model:"""

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
