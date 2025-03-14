from keras import Sequential
from keras.datasets import mnist
import numpy
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
import os
import sys
import os

# 获取根目录路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# 将根目录加入sys.path
sys.path.insert(0, project_root)
from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

(x_tr, y_tr), (x_te, y_te) = mnist.load_data()
print(x_tr.shape)

X_train = numpy.array([[1] * 128] * (10 ** 4) + [[0] * 128] * (10 ** 4))
X_test = numpy.array([[1] * 128] * (10 ** 2) + [[0] * 128] * (10 ** 2))

Y_train = numpy.array([True] * (10 ** 4) + [False] * (10 ** 4))
Y_test = numpy.array([True] * (10 ** 2) + [False] * (10 ** 2))

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

Y_train = Y_train.astype("bool")
Y_test = Y_test.astype("bool")

model = Sequential()
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('softmax'))
rms = RMSprop()

current_file_path = os.path.abspath(__file__)


# model.compile(loss='binary_crossentropy', optimizer=rms)
# model.fit(X_train, Y_train, batch_size=24, epochs=5, verbose=2, validation_data=(X_test, Y_test))
'''监测神经网络在训练过程中的状态
'''
dataset = {
    'x': X_train,
    'y': Y_train,
    'x_val': X_test,
    'y_val': Y_test
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=rms,loss='binary_crossentropy',dataset=dataset,iters=5,batch_size=24,callbacks=[],verb=2,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_test)
numpy.save(Y_test_name+'.npy', Y_test)
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
