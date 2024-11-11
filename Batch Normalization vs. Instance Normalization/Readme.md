In this experiment, we’ll compare **Batch Normalization** and **Instance Normalization** by training two models with the same architecture, differing only in their use of these normalization techniques. This comparison will help understand how each normalization method affects model training and performance.

### Experiment Steps

1. **Data Preparation**:
   - Load and preprocess the **CIFAR-10** dataset, which contains 32x32 color images in 10 classes.

2. **Define Model Architectures**:
   - Build two identical convolutional neural networks (CNNs), one with Batch Normalization and the other with Instance Normalization.
   
3. **Training and Comparison**:
   - Train each model and compare training and validation accuracies.

4. **Visualization**:
   - Plot the training and validation accuracies and losses over epochs for both models.

### Code Implementation

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Layer
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load and preprocess the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Custom Layer for Instance Normalization
class InstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=(input_shape[-1],),
                                     initializer="ones", trainable=True)
        self.beta = self.add_weight(name='beta', shape=(input_shape[-1],),
                                    initializer="zeros", trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        normalized = (x - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

# Function to build a CNN model with either BatchNorm or InstanceNorm
def build_model(normalization='batch'):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization() if normalization == 'batch' else InstanceNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization() if normalization == 'batch' else InstanceNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization() if normalization == 'batch' else InstanceNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Compile and train models
def train_model(model, epochs=10):
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_test, y_test))
    return history

# Build and train models with BatchNorm and InstanceNorm
batch_norm_model = build_model(normalization='batch')
instance_norm_model = build_model(normalization='instance')

print("Training model with Batch Normalization...")
batch_norm_history = train_model(batch_norm_model, epochs=10)

print("\nTraining model with Instance Normalization...")
instance_norm_history = train_model(instance_norm_model, epochs=10)

# Plot training and validation accuracy
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(batch_norm_history.history['accuracy'], label='BatchNorm Train')
plt.plot(batch_norm_history.history['val_accuracy'], label='BatchNorm Val')
plt.plot(instance_norm_history.history['accuracy'], label='InstanceNorm Train')
plt.plot(instance_norm_history.history['val_accuracy'], label='InstanceNorm Val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(batch_norm_history.history['loss'], label='BatchNorm Train')
plt.plot(batch_norm_history.history['val_loss'], label='BatchNorm Val')
plt.plot(instance_norm_history.history['loss'], label='InstanceNorm Train')
plt.plot(instance_norm_history.history['val_loss'], label='InstanceNorm Val')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
```

### Explanation

1. **Instance Normalization Layer**:
   - We define a custom `InstanceNormalization` layer, which calculates the mean and variance across spatial dimensions independently for each sample and channel. It includes trainable parameters, `gamma` and `beta`, to scale and shift the normalized output.

2. **Model Architecture**:
   - We build a simple CNN with alternating convolutional and normalization layers (Batch Normalization or Instance Normalization) followed by a dense output layer.
   
3. **Training**:
   - Each model is trained with 10 epochs, using `Adam` optimizer and categorical cross-entropy loss.
   - **Batch Normalization** normalizes inputs over a batch, while **Instance Normalization** normalizes within each sample, making it suitable for cases where each input needs independent normalization.

4. **Visualization**:
   - The training and validation accuracies and losses are plotted to compare model performance across epochs.

### Expected Results and Observations

- **Batch Normalization** generally stabilizes training by reducing internal covariate shift, which often leads to better accuracy and faster convergence.
- **Instance Normalization** is less effective in CNNs for image classification (unless the task benefits from per-instance normalization, like style transfer), but it may still show comparable or stable performance.

This experiment demonstrates how normalization choices affect model training and performance.