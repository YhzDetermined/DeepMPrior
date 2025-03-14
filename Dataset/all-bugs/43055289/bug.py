import numpy as np
from keras import Sequential
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense
from sklearn.datasets import make_classification
import os



X, y = make_classification(60, 225 * 225 * 3)
X = X.reshape(-1, 225, 225, 3)

# train_tensors, train_labels contain training data
model = Sequential()
model.add(Conv2D(filters=5,
                                  kernel_size=[4, 4],
                                  strides=2,
                                  padding='same',
                                  input_shape=[225, 225, 3]))
model.add(LeakyReLU(0.2))

model.add(Conv2D(filters=10,
                                  kernel_size=[4, 4],
                                  strides=2,
                                  padding='same'))
model.add(BatchNormalization(axis=3))
model.add(LeakyReLU(0.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

model.fit(X, y, batch_size=8, epochs=5, shuffle=True)

metrics = model.evaluate(X, y)
print('')
print(np.ravel(model.predict(X)))
print('training data results: ')
for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i]) + ": " + str(metrics[i]))

def get_dense_weights(model):
    """
    提取模型中 Dense 层的权重和偏置。

    参数：
    - model: Keras 的 Sequential 或 Functional 模型对象。

    返回：
    - dense_weights: 一个列表，包含所有 Dense 层的权重和偏置。
      格式：[weights_1, biases_1, weights_2, biases_2, ...]
    """
    dense_weights = []
    for layer in model.layers:
        if isinstance(layer, Dense):  # 检查是否是 Dense 层
            dense_weights.extend(layer.get_weights())  # 获取权重和偏置
    return dense_weights
weights = get_dense_weights(model)
for i, w in enumerate(weights):
    print(f"Weight {i+1} shape: {w.shape}")

