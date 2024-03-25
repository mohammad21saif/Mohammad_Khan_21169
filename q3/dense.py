from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from itertools import product


# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))

# Scaling
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

#One hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


def build_model(hidden_layers=1, units=512, learning_rate=0.001, dropout_rate=0.0, activation='relu'):
    model = Sequential()
    model.add(Dense(units, activation=activation, input_shape=(784,)))
    model.add(Dropout(dropout_rate))
    for _ in range(hidden_layers - 1):
        model.add(Dense(units, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Trying out parameters
parameters = {
    'hidden_layers': [1, 2],
    'units': [128, 256, 512],
    'learning_rate': [0.001, 0.003],
    'dropout_rate': [0.0, 0.2],
    'activation': ['relu', 'tanh']
}


for params in product(*parameters.values()):
    config = dict(zip(parameters.keys(), params))
    print("Testing config: ", config)
    model = build_model(**config)
    model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_split=0.2, verbose=0)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}\n")

