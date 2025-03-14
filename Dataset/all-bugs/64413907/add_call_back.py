import numpy
from keras.layers import LSTM, Masking, Dense
from keras.utils import to_categorical
from keras import models, losses
import tensorflow as tf
import numpy as np
import os

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

"""
For generating reproducible results, set seed. 
"""


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


"""
Set some right most indices to mask value like padding
"""


def create_padded_seq(num_samples, timesteps, num_feats, mask_value):
    feats = np.random.random([num_samples, timesteps, num_feats]).astype(np.float32)  # Generate samples
    for i in range(0, num_samples):
        rand_index = np.random.randint(low=2, high=timesteps, size=1)[0]  # Apply padding
        feats[i, rand_index:, 0] = mask_value
    return feats


set_seed(42)
num_samples = 100
timesteps = 6
num_feats = 1
num_classes = 3
num_lstm_cells = 1
mask_value = -100
num_epochs = 5

X_train = create_padded_seq(num_samples, timesteps, num_feats, mask_value)
y_train = np.random.randint(num_classes, size=num_samples)
cat_y_train = to_categorical(y_train, num_classes)

masked_model = models.Sequential(name='masked')
masked_model.add(Masking(mask_value=mask_value, input_shape=(timesteps, num_feats)))
masked_model.add(LSTM(num_lstm_cells, return_sequences=False))
masked_model.add(Dense(num_classes, activation='relu'))
#masked_model.compile(loss=losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
print(masked_model.summary())
#masked_model.fit(X_train, cat_y_train, batch_size=1, epochs=5, verbose=True)
dataset = {
    'x': X_train,
    'y': cat_y_train,
    'x_val': X_train,
    'y_val': cat_y_train
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=masked_model,optimizer='adam',loss=losses.categorical_crossentropy,dataset=dataset,iters=5,batch_size=1,callbacks=[],verb=True,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', X_train)
numpy.save(Y_test_name+'.npy', cat_y_train)