import os

import numpy
from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

from feature_extraction.Config import train_model_name, X_test_name, Y_test_name, params
from feature_extraction.Util.Utils import my_model_train

train_x, train_y = make_classification(1000, 10 * 150, random_state=42)
train_x = train_x.reshape(-1, 10, 150)

model = Sequential()

model.add(Dense(32, input_shape=(len(train_x[0]), 150), activation="relu"))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Flatten())
model.add(Dense(1, activation="softmax"))

#model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#hist = model.fit(train_x, train_y, epochs=200, batch_size=2, verbose=1)

dataset = {
    'x': train_x,
    'y': train_y,
    'x_val': train_x,
    'y_val': train_y
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer="adam",loss='binary_crossentropy',dataset=dataset,iters=200,batch_size=2,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', train_x)
numpy.save(Y_test_name+'.npy', train_y)