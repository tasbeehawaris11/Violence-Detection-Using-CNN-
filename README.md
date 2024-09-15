## Image Classification with Data Augmentation and Training History (README.md)

This project implements an image classification model using TensorFlow and Keras. It demonstrates data augmentation techniques and saving the training history for later analysis.

### Functionality

1. **Data Preparation:**
    * Loads image datasets from directories (train, test, validation)
    * Converts images into arrays and resizes them
    * Visualizes sample images with corresponding class labels
2. **Data Augmentation:**
    * Applies random image transformations (flipping, rotation, zoom, contrast) to increase training data variety
3. **Model Architecture:**
    * Uses a Convolutional Neural Network (CNN) with:
        * Rescaling layer for normalization
        * Convolutional layers with ReLU activation
        * MaxPooling layers for dimensionality reduction
        * Flatten layer to convert 2D feature maps to 1D vectors
        * Dropout layer to prevent overfitting
        * Dense layers with ReLU activation for classification
        * Softmax activation layer for multi-class classification
4. **Training:**
    * Compiles the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
    * Trains the model with early stopping and learning rate reduction callbacks
    * Saves the best model based on validation loss
5. **Evaluation:**
    * Saves the training history for plotting accuracy and loss curves

### Requirements

* Python 3.x
* TensorFlow
* Keras
* Matplotlib (for visualization)
* Pickle (for saving training history)

### Usage

1. Update the script with your specific dataset paths (train, test, validation) in the `data_*_path` variables.
2. Adjust image dimensions (`img_width` and `img_height`) if needed.
3. Modify the model architecture and hyperparameters (epochs, learning rate) based on your dataset.
4. Run the script:

```bash
python your_script_name.py
```

**Output:**

* Saved models: `best_model.keras`, `model.keras`
* Saved training history: `history.pkl`
* Training history plot: `training_history.png`

### Disclaimer

This script serves as a starting point for image classification tasks. You might need to adapt it to your specific dataset and desired functionalities.
# Violence-Detection-Using-CNN-
