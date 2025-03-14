import os

import numpy
from keras import Sequential
from keras.datasets.cifar10 import load_data
from keras.layers import Dense, Flatten
from keras.utils import to_categorical, normalize

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用 GPU，强制使用 CPU
(x_train, y_train), (_, _) = load_data()
x_train = normalize(x_train)
y_train = to_categorical(y_train)
model = Sequential()
model.add(Flatten())
model.add(Dense(units=8, activation='sigmoid'))
model.add(Dense(units=16, activation='sigmoid'))
model.add(Dense(units=32, activation='sigmoid'))
model.add(Dense(units=64, activation='sigmoid'))
model.add(Dense(units=128, activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=1000, batch_size=500)
dataset = {
    'x': x_train,
    'y': y_train,
    'x_val': x_train,
    'y_val': y_train
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='categorical_crossentropy',dataset=dataset,iters=1000,batch_size=500,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', x_train)
numpy.save(Y_test_name+'.npy', y_train)