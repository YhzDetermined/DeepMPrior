import os

import numpy
from keras import layers, models
from tensorflow import keras

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

data = [[1., 1.], [1., 0.], [0., 1.], [0., 0.]]
results = [[1.], [0.], [0.], [0.]]


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(len(data[0]), activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))

#    model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.Accuracy()], optimizer='adam')

    return model


model = build_model()

#model.fit(data, results, epochs=1000)

dataset = {
    'x': data,
    'y': results,
    'x_val': data,
    'y_val': results
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='binary_crossentropy',dataset=dataset,iters=1000,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', data)
numpy.save(Y_test_name+'.npy', results)



model.summary()


print(model.predict([data[0]]))
print(model.predict([data[1]]))
print(model.predict([data[2]]))
print(model.predict([data[3]]))
