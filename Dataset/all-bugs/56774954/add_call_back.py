import os

import numpy
from keras import Sequential, Model, Input
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.datasets import make_regression

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

m = 10
n = 3

X, y = make_regression(1000, n_features=m, n_targets=n)

inputs = Input(shape=(m,))
hidden = Dense(100, activation='sigmoid')(inputs)
hidden = Dense(80, activation='sigmoid')(hidden)
outputs = Dense(n, activation='softmax')(hidden)

opti = Adam(learning_rate=0.001)

model = Model(inputs=inputs, outputs=outputs)
# model.compile(optimizer=opti,
#               loss='poisson',
#               metrics=['accuracy'])
#
# model.fit(X, y, verbose=2, batch_size=32, epochs=30)
dataset = {
    'x': X,
    'y': y,
    'x_val': X,
    'y_val': y
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=opti,loss='poisson',dataset=dataset,iters=30,batch_size=32,callbacks=[],verb=2,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X)
numpy.save(Y_test_name+'.npy', y)