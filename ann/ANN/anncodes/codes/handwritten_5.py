import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import cv2  # OpenCV for image processing
import matplotlib.pyplot as plt

# Step 1: Load and preprocess MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize

# Step 2: Define and Train the Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
print("\nModel training complete!")

# Step 3: Function to Load and Predict on an External Image
def predict_digit(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    if img is None:
        print("Error: Image not found!")
        return
    
    # Resize to 28x28 (same as MNIST)
    img = cv2.resize(img, (28, 28))
    
    # Invert colors if needed (MNIST digits are white on black)
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # Normalize pixel values (same as MNIST preprocessing)
    img = img / 255.0
    
    # Reshape to match model input (1 sample, 28x28)
    img = img.reshape(1, 28, 28)
    
    # Predict using the trained model
    prediction = model.predict(img)
    
    # Get the predicted class
    predicted_label = np.argmax(prediction)
    
    # Display the image and prediction
    plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"Predicted Number: {predicted_label}")
    plt.axis('off')
    plt.show()
    
    print(f"Model Prediction: {predicted_label}")

# Step 4: Use the function to predict a digit from an image
image_path = "20250318_150100.jpg"  # Change to your image path
predict_digit(image_path)
