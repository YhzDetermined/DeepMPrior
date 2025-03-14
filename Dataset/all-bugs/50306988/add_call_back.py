import os

import numpy
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

dataset_size = 200

class_1 = np.random.uniform(low=0.2, high=0.4, size=(dataset_size,))
class_2 = np.random.uniform(low=0.7, high=0.5, size=(dataset_size,))

dataset = []
for i in range(0, dataset_size, 2):
    dataset.append([class_1[i], class_1[i + 1], 1])
    dataset.append([class_2[i], class_2[i + 1], 2])

df_train = pd.DataFrame(data=dataset, columns=['x', 'y', 'class'])

x_train = df_train.iloc[:, 0:-1].values
y_train = df_train.iloc[:, -1]

nr_feats = x_train.shape[1]
nr_classes = y_train.nunique()

label_enc = LabelEncoder()
label_enc.fit(y_train)

y_train = to_categorical(label_enc.transform(y_train), nr_classes)

model = Sequential()
model.add(Dense(units=2, activation='sigmoid', input_shape=(nr_feats,)))
model.add(Dense(units=nr_classes, activation='softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=True)
dataset = {
    'x': x_train,
    'y': y_train,
    'x_val': x_train,
    'y_val': y_train
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer='adam',loss='categorical_crossentropy',dataset=dataset,iters=500,batch_size=32,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', x_train)
numpy.save(Y_test_name+'.npy', y_train)
