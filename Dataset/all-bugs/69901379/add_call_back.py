import os

import numpy
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

n = 50
x = np.random.randint(50, 2000, (n, 10))
y = np.random.randint(600, 4000, (n, 1))

k = 16

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(10,)),
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(1)
])

#model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))
#history = model.fit(x, y, epochs=1000, batch_size=32, verbose=0)

dataset = {
    'x': x,
    'y': y,
    'x_val': x,
    'y_val': y
}
check_interval='epoch_2'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=Adam(learning_rate=0.0001),loss='mse',dataset=dataset,iters=1000,batch_size=32,callbacks=[],verb=0,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', x)
numpy.save(Y_test_name+'.npy', y)


# loss = history.history['loss']
#
# for epoch in [1, 10, 50, 100, 500, 1000]:
#     print('Epoch: {}, Loss: {:,.4f}'.format(epoch, loss[epoch - 1]))
