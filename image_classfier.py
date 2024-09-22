import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from tensorflow.keras import datasets, layers, models

# Load and normalize data
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Class names
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Plot the first 16 images with labels
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

# Reducing the dataset size for quick testing
training_images = training_images[:2000]
training_labels = training_labels[:2000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Model definition
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # Corrected input shape
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# # Evaluate the model
# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

# # Save the model in .keras format
# model.save('image_classifier.keras')

# Load the saved model for prediction
model = models.load_model('image_classifier.keras')

# Load an image for prediction
img = cv.imread('Image_classifier\FrogBlur.jpg')

# Ensure the image is found and loaded
if img is None:
    print("Error: Image not found or unable to open.")
else:
    # Convert the image to RGB format
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()

    # Preprocess the image before prediction
    img_resized = cv.resize(img, (32, 32))  # Resize to match training images size
    img_resized = np.array([img_resized]) / 255.0  # Normalize the image

    # Make a prediction
    prediction = model.predict(img_resized)
    index = np.argmax(prediction)
    print(f"Prediction is: {class_names[index]}")
