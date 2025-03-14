import os

import numpy
import tensorflow as tf
import random
from sklearn.datasets import make_classification

from feature_extraction.Config import train_model_name, X_test_name, Y_test_name, params
from feature_extraction.Util.Utils import my_model_train

X, y = make_classification(1000, 31, random_state=42)

model = tf.keras.Sequential()

num_nodes = [1]
act_functions = [tf.nn.relu]
optimizers = ['SGD']
loss_functions = ['binary_crossentropy']
epochs_count = ['10']
batch_sizes = ['500']

act = random.choice(act_functions)
opt = random.choice(optimizers)
ep = random.choice(epochs_count)
batch = random.choice(batch_sizes)
loss = random.choice(loss_functions)
count = random.choice(num_nodes)

model.add(tf.keras.layers.Dense(31, activation=act, input_shape=(31,)))
model.add(tf.keras.layers.Dense(count, activation=act))
model.add(tf.keras.layers.Dense(1, activation=act))
#model.compile(loss=loss,
#              optimizer=opt,
#              metrics=['accuracy'])


epochs = int(ep)
batch_size = int(batch)
#model.fit(X, y, epochs=epochs, batch_size=batch_size)

dataset = {
    'x': X,
    'y': y,
    'x_val': X,
    'y_val': y
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=opt,loss=loss,dataset=dataset,iters=epochs,batch_size=batch_size,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)