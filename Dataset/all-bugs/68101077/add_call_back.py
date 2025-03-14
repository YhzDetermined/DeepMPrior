import os
import time
from keras import layers
import keras
import numpy
from keras.datasets.fashion_mnist import load_data

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

(train_images, train_labels), (test_images, test_labels) = load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# model.compile(optimizer='adam',
#               loss="sparse_categorical_crossentropy",
#               metrics=['accuracy'])
#
# model.fit(train_images, train_labels, epochs=10)
dataset = {
    'x': train_images,
    'y': train_labels,
    'x_val': test_images,
    'y_val': test_labels
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss="sparse_categorical_crossentropy",dataset=dataset,iters=10,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', test_images)
numpy.save(Y_test_name+'.npy', test_labels)
