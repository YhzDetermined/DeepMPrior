import os

import numpy
from keras import Input, applications, Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from sklearn.datasets import make_classification

from feature_extraction.Config import train_model_name, X_test_name, Y_test_name, params
from feature_extraction.Util.Utils import my_model_train

img_width, img_height = 150, 150
nb_train_samples = 200

x, y = make_classification(nb_train_samples, img_width * img_height * 3, random_state=42)
x = x.reshape(-1, 150, 150, 3)

input_tensor = Input(shape=(150, 150, 3))

base_model = applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
for layer in base_model.layers:
    layer.trainable = False

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation="relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='softmax'))
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.fit(x, y, epochs=50, batch_size=64)
dataset = {
    'x': x,
    'y': y,
    'x_val':x,
    'y_val': y
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='binary_crossentropy',dataset=dataset,iters=50,batch_size=64,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', x)
numpy.save(Y_test_name+'.npy', y)