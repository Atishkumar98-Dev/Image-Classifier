import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split

def load_data(data_dir, image_size=(32, 32)):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))  # Get class names from directory structure
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv.imread(img_path)
            if img is not None:
                img = cv.resize(img, image_size)
                images.append(img)
                labels.append(label)
    
    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels)
    
    return images, labels, class_names

# Load and preprocess your dataset
data_dir = 'F:\ML_project\Image_classifier\data_directory'  # Update this path
images, labels, class_names = load_data(data_dir)

# Split into training and testing datasets
training_images, testing_images, training_labels, testing_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Model definition
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(class_names), activation='softmax'))  # Output layer matches number of classes

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# Save the model
model.save('Image_classifier\custom_image_classifier.keras')

# Load an image for prediction
img = cv.imread('Image_classifier\DeerBlur.jpg')  # Update this path

# Ensure the image is found and loaded
if img is None:
    print("Error: Image not found or unable to open.")
else:
    # Convert the image to RGB format
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Resize and preprocess the image for prediction
    img_resized = cv.resize(img, (32, 32))  # Resize to match training images size
    img_resized = np.array([img_resized]) / 255.0  # Normalize the image
    
    # Make a prediction
    prediction = model.predict(img_resized)
    index = np.argmax(prediction)
    print(f"Prediction is: {class_names[index]}")
    
    # Optionally display the image
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(f'Predicted: {class_names[index]}')
    plt.axis('off')
    plt.show()
