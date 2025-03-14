import os

import keras
import numpy
import numpy as np

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

input1 = np.array([[2], [1], [4], [3], [5]])
input2 = np.array([[2, 1, 8, 4], [2, 6, 1, 9], [7, 3, 1, 4], [3, 1, 6, 10], [3, 2, 7, 5]])
outputs = np.array([[3, 3, 1, 0], [3, 3, 3, 0], [3, 3, 4, 0], [3, 3, 1, 0], [3, 3, 4, 0]])

merged = np.column_stack([input1, input2])
model = keras.Sequential([
    keras.layers.Dense(2, input_dim=5, activation='relu'),
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(4, activation='sigmoid'),
])

#model.compile(
#    loss="mean_squared_error", optimizer="adam", metrics=["accuracy"]
#)

#model.fit(merged, outputs, batch_size=16, epochs=100)
dataset = {
    'x': merged,
    'y': outputs,
    'x_val': merged,
    'y_val': outputs
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer="adam",loss="mean_squared_error",dataset=dataset,iters=100,batch_size=16,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', merged)
numpy.save(Y_test_name+'.npy', outputs)