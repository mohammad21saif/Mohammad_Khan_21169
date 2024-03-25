from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from itertools import product

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))

# Scaling
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# One hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def build_model(hidden_layers, units1, units2, learning_rate, dropout_rate, activation):
    model = Sequential()
    model.add(Dense(units1, activation=activation, input_shape=(784,)))
    model.add(Dropout(dropout_rate))
    if hidden_layers > 1 and units2 is not None:
        model.add(Dense(units2, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

parameters = {
    'hidden_layers': [1, 2],
    'units1': [128, 256, 512],
    'units2': [128, 256, 512],
    'learning_rate': [0.001, 0.003],
    'dropout_rate': [0.0, 0.2],
    'activation': ['relu', 'tanh']
}

best_acc = 0.0
best_config = None

for params in product(*parameters.values()):
    hidden_layers, units1, units2, learning_rate, dropout_rate, activation = params
    config = {
        'hidden_layers': hidden_layers,
        'units1': units1,
        'units2': units2 if hidden_layers > 1 else None,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'activation': activation
    }
    model = build_model(**config)
    model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_split=0.2, verbose=0)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_config = config
    
    print(f"Testing config: {config}, Test accuracy: {test_acc:.4f}")

print("\nBest configuration:")
print(best_config)
print(f"Best test accuracy: {best_acc:.4f}")
