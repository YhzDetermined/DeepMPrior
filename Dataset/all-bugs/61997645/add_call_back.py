import os

import numpy
from keras.layers import GRU, Dropout, Dense
from keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

data_len = 300
x = []
y = []
for i in range(data_len):
    a = np.random.randint(1,10,5)
    if a[0] % 2 == 0:
        y.append('0')
    else:
        y.append('1')

    a = a.reshape(5, 1)
    x.append(a)
    # print(x)

X = np.array(x)
Y = np.array(y)

model = Sequential()
model.add(GRU(units=24, activation='relu', return_sequences=True, input_shape=[5,1]))
model.add(Dropout(rate=0.5))
model.add(GRU(units=12, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=1, activation='softmax'))
#
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# model.summary()
# history = model.fit(X[:210], Y[:210], epochs=20, validation_split=0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X[:210], Y[:210], test_size=0.2, random_state=42)
dataset = {
    'x': X_train,
    'y': Y_train,
    'x_val': X_val,
    'y_val': Y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='mse',dataset=dataset,iters=20,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X[:210])
numpy.save(Y_test_name+'.npy', Y[:210])
