import numpy
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

from feature_extraction.Config import params, train_model_name, X_test_name, Y_test_name
from feature_extraction.Util.Utils import my_model_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def abc(x1, x2):
    b = -2 * x1 * x2
    c = x1 * x2
    sol = [b, c]
    return sol


a = 10
b = 10
c = a * b


def Nx2(N, M):
    matrix = []
    n = N + 1
    m = M + 1
    for i in range(1, n):
        for j in range(1, m):
            temp = [i, j]
            matrix.append(temp)
    final_matrix = np.array(matrix)
    return final_matrix


output = Nx2(a, b)

# print(output)

input = []
for i in range(0, c):
    temp2 = abc(output[i, 0], output[i, 1])
    input.append(temp2)
input = np.array(input)

print(input)

train_labels = output
train_samples = input

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
scaled_train_samples = scaled_train_samples.reshape(-1, 2)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_labels = scaler.fit_transform(train_labels.reshape(-1, 1))
scaled_train_labels = scaled_train_labels.reshape(-1, 2)

print(scaled_train_samples)
print(scaled_train_labels)

model = Sequential([
    Dense(2, input_shape=(2,), activation='sigmoid'),
    Dense(2, activation='sigmoid'),
])

# print(model.weights)
#
# model.compile(SGD(learning_rate=0.01), loss='mean_squared_error', metrics=['accuracy'])
# model.fit(scaled_train_labels, scaled_train_labels, validation_split=0.2, batch_size=10, epochs=20, shuffle=True,
#           verbose=2)
#
X_train, X_val, Y_train, Y_val = train_test_split(scaled_train_labels, scaled_train_labels, test_size=0.2, random_state=42)
dataset = {
    'x': X_train,
    'y': Y_train,
    'x_val': X_val,
    'y_val': Y_val
}
check_interval='epoch_1'
save_dir = os.path.join("result_dir")
log_dir = os.path.join("log_dir")
res,model=my_model_train(model=model,optimizer=SGD(learning_rate=0.01),loss='mean_squared_error',dataset=dataset,iters=20,batch_size=10,callbacks=[],verb=1,save_dir=save_dir, determine_threshold=1, params=params,
               checktype=check_interval,log_dir=log_dir)
model.save(train_model_name+".h5")
numpy.save(X_test_name+'.npy', scaled_train_labels)
numpy.save(Y_test_name+'.npy', scaled_train_labels)
print(model.summary())
print(model.weights)

