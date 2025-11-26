# Implement a Transfer Learning concept in Image Classification

## AIM
To develop an image classification model using transfer learning by using the pre-trained MobileNetV2 architecture and training a new classifier on the CIFAR-10 dataset.

## THEORY

Transfer Learning is a deep learning technique where a model trained on a large dataset is reused for another related task. Instead of training a neural network from scratch, we use a pre-trained model such as VGG19, ResNet, or MobileNetV2, which already learned powerful feature representations from millions of images. This reduces training time, improves accuracy, and requires less data.

MobileNetV2 is an efficient convolutional neural network designed for mobile and embedded systems. It uses depthwise separable convolutions, inverted residual blocks, and linear bottlenecks. These techniques help reduce the number of parameters while still maintaining good accuracy. In transfer learning, the base layers of MobileNetV2 act as a feature extractor, and only the top layers (classifier) are newly added and trained. For the CIFAR-10 dataset, new dense layers are attached at the end to classify the 10 image categories.


## DESIGN STEPS

STEP 1: Import the required libraries

Load TensorFlow and Keras modules needed for building and training the neural network.

STEP 2: Load and preprocess the CIFAR-10 dataset

Load the training and testing images.
Normalize the pixel values to the range 0–1 so the model learns efficiently.

STEP 3: Load the pre-trained MobileNetV2 model

Use MobileNetV2 with ImageNet weights.
Remove the top classifier layer since we will add our own.
Freeze the base model so its weights do not change during training.

STEP 4: Add custom classification layers

Attach layers like Global Average Pooling, Dense layer, Dropout, and final output layer for 10 classes.
These layers will learn to classify the CIFAR-10 images.

STEP 5: Compile the model

Choose an optimizer (Adam), loss function (cross-entropy), and accuracy as the performance metric.

STEP 6: Train the model

Train the model for a few epochs using the training data and validate on the test data.

STEP 7: Evaluate the model

Measure the accuracy and loss of the trained model on unseen test images.

STEP 8: Save the trained model

Store the final trained model in a file so it can be loaded and used later.

## PROGRAM

## NAME:BASHA VENU

## REG NO:2305001005
``` python
# Experiment: Transfer Learning for Image Classification
# ------------------------------------------------------

# Step 1: Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 2: Load and Preprocess Dataset (CIFAR-10)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to range [0,1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Step 3: Load Pre-trained Model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base layers

# Step 4: Add Custom Classification Layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # 10 classes in CIFAR-10
])

# Step 5: Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(
    train_images, train_labels,
    epochs=5,
    validation_data=(test_images, test_labels)
)

# Step 7: Evaluate the Model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Step 8: Save the Model
model.save("transfer_learning_model.h5")
print("\nModel saved successfully!")


```

## OUTPUT

<img width="1220" height="237" alt="image" src="https://github.com/user-attachments/assets/7e410507-1411-482f-b51f-5e37ca2b563e" />
<img width="270" height="77" alt="image" src="https://github.com/user-attachments/assets/583a949c-19e9-4e7f-91a2-2a72d00fab09" />


**RESULT**

Thus, the transfer learning model using MobileNetV2 was successfully implemented, trained, evaluated on the CIFAR-10 dataset.
