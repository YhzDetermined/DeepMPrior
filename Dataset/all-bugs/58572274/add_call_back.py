import os

from keras.models import Sequential
from keras.layers import Dense
import numpy

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

data = numpy.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]])
train = data[:, :-1]  # Taking The same and All data for training
test = data[:, :-1]

train_l = data[:, -1]
test_l = data[:, -1]

train_label = []
test_label = []

for i in train_l:
    train_label.append([i])
for i in test_l:
    test_label.append([i])  # Just made Labels Single element...

train_label = numpy.array(train_label)
test_label = numpy.array(test_label)  # Numpy Conversion

model = Sequential()

model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))

# model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer='adam')
#
# model.fit(train, train_label, epochs=10, verbose=2)
dataset = {
    'x': train,
    'y': train_l,
    'x_val': test,
    'y_val': test_l
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='binary_crossentropy',dataset=dataset,iters=10,batch_size=32,callbacks=[],verb=2,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', test)
numpy.save(Y_test_name+'.npy', test_l)

model.predict_classes(test)
