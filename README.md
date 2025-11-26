# Implement a Transfer Learning concept in Image Classification

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## THEORY

Neural Network Model

<img width="937" height="332" alt="image" src="https://github.com/user-attachments/assets/447092a0-3278-4df0-bcdb-db21021377db" />


## DESIGN STEPS

STEP 1: We begin by importing the necessary Python libraries, including TensorFlow for deep learning, data preprocessing tools, and visualization libraries.

STEP 2: To leverage the power of GPU acceleration, we configure TensorFlow to allow GPU processing, which can significantly speed up model training.

STEP 3: We load the dataset, consisting of cell images, and check their dimensions. Understanding the image dimensions is crucial for setting up the neural network architecture.

STEP 4: We create an image generator that performs data augmentation, including rotation, shifting, rescaling, and flipping. Data augmentation enhances the model's ability to generalize and recognize malaria-infected cells in various orientations and conditions. 

STEP 5: We design a convolutional neural network (CNN) architecture consisting of convolutional layers, max-pooling layers, and fully connected layers. The model is compiled with appropriate loss and optimization functions.

STEP 6: We split the dataset into training and testing sets, and then train the CNN model using the training data. The model learns to differentiate between parasitized and uninfected cells during this phase.

STEP 7: We visualize the training and validation loss to monitor the model's learning progress and detect potential overfitting or underfitting.

STEP 8: We evaluate the trained model's performance using the testing data, generating a classification report and confusion matrix to assess accuracy and potential misclassifications.

STEP 9: We demonstrate the model's practical use by randomly selecting and testing a new cell image for classification.

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

The model's performance is evaluated through training and testing, and it shows potential for assisting healthcare professionals in diagnosing malaria more efficiently and accurately
