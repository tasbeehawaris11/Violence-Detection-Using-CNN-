import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Dataset paths
data_train_path = r"C:\Users\acer\Downloads\archive (3)\train"
data_test_path = r"C:\Users\acer\Downloads\archive (3)\test"
data_val_path = r"C:\Users\acer\Downloads\archive (3)\val"

img_width = 180
img_height = 180

# Converting data into arrays
dataset = image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=32,
)

data_cat = dataset.class_names
print(data_cat)

test_dataset = image_dataset_from_directory(
    data_test_path,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=32,
)

val_dataset = image_dataset_from_directory(
    data_val_path,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=32,
)

# Visualize sample images
def visualize_samples(dataset, class_names, num_samples=9):
    plt.figure(figsize=(10, 10))
    for image, labels in dataset.take(1):
        for i in range(num_samples):
            plt.subplot(3, 3, i + 1)
            plt.imshow(image[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

visualize_samples(dataset, data_cat)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# Model architecture
model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(data_cat), activation='softmax')
])

# Learning rate schedule# Initialize optimizer with a fixed learning rate
# Initialize optimizer with a fixed learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

# Training parameters
epochs = 20

# Callbacks
callbacks = [
    ModelCheckpoint(filepath=r"C:\Users\acer\Downloads\best_model.keras", save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=5),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

# Train the model
history = model.fit(
    dataset,
    validation_data=val_dataset,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks
)

# Save the model
model.save(r"C:\Users\acer\Downloads\model.keras")

# Save the training history
with open(r'C:\Users\acer\Downloads\history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# Load the training history
with open(r'C:\Users\acer\Downloads\history.pkl', 'rb') as file:
    history = pickle.load(file)

# Plot accuracy and loss
def plot_history(history, save_path):
    plt.figure(figsize=(12, 6))

    # Accuracy subplot
    if 'accuracy' in history and 'val_accuracy' in history:
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')

    # Loss subplot
    if 'loss' in history and 'val_loss' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over Epochs')

    # Save the figure
    plt.savefig(save_path)  # Save as PNG file
    plt.show()  # Display the plot

plot_history(history, r'C:\Users\acer\Downloads\training_history.png')



