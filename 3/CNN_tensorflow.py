import pandas as pd
import tensorflow as tf
print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt

(train_features,train_labels),(test_features,test_labels)=tf.keras.datasets.mnist.load_data()
train_features = train_features/255
test_features = test_features/255
print("训练集个数与照片尺寸 {}".format(train_features.shape))
print("测试集个数与照片尺寸 {}".format(test_features.shape))

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Reshape([28,28,1]),
    tf.keras.layers.Conv2D(32, (7, 7), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(10, activation='softmax')


])

model.summary()

"""模型编译"""
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['acc']
)

"""模型训练"""
model.fit(train_features, train_labels , epochs = 10, batch_size=1000)
"""模型评价"""
model.evaluate(test_features, test_labels, verbose=2)