import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2

# Step 1: Load MNIST data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocess the images: Normalize and flatten them
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255.0

# Step 2: Build a simple DNN model
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28, 1)))  # Flatten the 28x28 image to a vector
model.add(layers.Dense(128, activation='relu'))  # Dense layer with 128 neurons
model.add(layers.Dense(10, activation='softmax'))  # Output layer with 10 classes (digits 0-9)

# Step 3: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Step 5: Save the trained model
save_path = ".TP10\\mnist_model.h5"
model.save(save_path)
print("Model saved to " + save_path)

# Step 6: Load the model in OpenCV
# Convert the TensorFlow model to OpenCV format (using DNN module in OpenCV)
# OpenCV DNN module uses models saved in the "TensorFlow SavedModel" or "TF Lite" formats

# First, we need to convert the model to a format that OpenCV can read:
# Use TensorFlow's SavedModel format
"""
model.save(".TP10\\mnist_saved_model")

# Step 7: Load the model in OpenCV
# OpenCV DNN module can load models in the TensorFlow SavedModel format
net = cv2.dnn.readNetFromTensorflow(".\\mnist_saved_model\\saved_model.pb")

# Step 8: Test the model with OpenCV
# Prepare an image for testing (let's take the first image from the test set)
image = test_images[0]

# Convert the image to a format suitable for OpenCV's DNN module (HWC, float32)
image = cv2.dnn.blobFromImage(image[0], 1.0, (28, 28), (0, 0, 0), swapRB=False, crop=False)

# Perform forward pass with OpenCV's DNN
net.setInput(image)
output = net.forward()

# Print the output (probabilities for each digit 0-9)
print("Output from OpenCV DNN forward pass: ", output)

# Get the predicted digit
predicted_digit = np.argmax(output)
print(f"Predicted digit: {predicted_digit}")
"""