import os

import numpy as np
from keras import Sequential
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense
from sklearn.datasets import make_classification

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train
# os.environ["PATH"] = r"C:\Users\27754\.conda\envs\tf3-gpu\Library\bin;" + os.environ["PATH"]
X, y = make_classification(60, 225 * 225 * 3)
X = X.reshape(-1, 225, 225, 3)

# train_tensors, train_labels contain training data
model = Sequential()
model.add(Conv2D(filters=5,
                                  kernel_size=[4, 4],
                                  strides=2,
                                  padding='same',
                                  input_shape=[225, 225, 3]))
model.add(LeakyReLU(0.2))

model.add(Conv2D(filters=10,
                                  kernel_size=[4, 4],
                                  strides=2,
                                  padding='same'))
model.add(BatchNormalization(axis=3))
model.add(LeakyReLU(0.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy',
#                             optimizer='adam',
#                             metrics=['accuracy'])
#
# model.fit(X, y, batch_size=8, epochs=5, shuffle=True)
dataset = {
    'x': X,
    'y': y,
    'x_val': X,
    'y_val': y
}
check_interval = 'epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model, optimizer='adam', loss='binary_crossentropy', dataset=dataset, iters=5, batch_size=8,
               callbacks=[], verb=1, save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval, log_dir=log_dir)
model.save(train_model_name+".h5")
np.save(X_test_name+'.npy', X)
np.save(Y_test_name+'.npy', y)
metrics = model.evaluate(X, y)
print('')
print(np.ravel(model.predict(X)))
print('training data results: ')
for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i]) + ": " + str(metrics[i]))
