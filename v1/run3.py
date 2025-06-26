import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1. Load and Preprocess the MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
# Example: 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# 2. Build the 4-Layer Neural Network Model
model = Sequential([
    # Input Layer: Flattens the 28x28 images into a 1D array of 784 pixels
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# 3. Compile the Model
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)


model.compile(optimizer=sgd_optimizer,  #  <----- Adam optimizer is a good default
              loss='categorical_crossentropy',  # Suitable for one-hot encoded labels
              metrics=['accuracy'])

model.summary()

# 4. Train the Model
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32, # Number of samples per gradient update
                    validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
