

# Vanishing and Exploding Gradient Experiment

This project demonstrates the vanishing and exploding gradient problem by training neural networks with different architectures and observing how gradients behave during training. We use the MNIST dataset and experiment with various network depths and activation functions to analyze how gradients behave in shallow versus deep networks.

## Table of Contents

- [Overview](#overview)
- [Experiment Setup](#experiment-setup)
- [Requirements](#requirements)
- [Code Details](#code-details)
- [Usage](#usage)
- [Results](#results)
- [Observations](#observations)
- [Conclusion](#conclusion)

## Overview

The vanishing and exploding gradient problem affects deep neural networks during backpropagation. Vanishing gradients occur when gradients become too small to update weights effectively, while exploding gradients result in overly large weights, causing instability. This experiment helps visualize and understand these issues by tracking gradients across epochs for networks with different depths and activation functions.

## Experiment Setup

1. **Data**: The MNIST dataset of handwritten digits.
2. **Network Architectures**:
   - **Shallow Network** (5 layers)
   - **Deep Network** (20 layers)
3. **Activation Functions**:
   - Sigmoid (prone to vanishing gradients)
   - ReLU (mitigates vanishing gradient issue but can lead to exploding gradients in deep networks)
4. **Gradient Tracking**:
   - Gradients are tracked layer-by-layer to monitor their values across training epochs.

## Requirements

This project requires the following libraries:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

Install dependencies using:
```bash
pip install tensorflow numpy matplotlib
```

## Code Details

The primary components of the code are:

1. **Data Loading and Preprocessing**:
   - The MNIST dataset is loaded and normalized to ensure stable training.

2. **Network Architectures**:
   - The `create_model()` function creates a neural network model with a specified number of layers and activation function.

3. **Gradient Tracking**:
   - The `GradientTracker` callback monitors the average gradient values at each epoch to help visualize gradient behavior.
   
4. **Experiment Configurations**:
   - We test four configurations: shallow and deep networks with both Sigmoid and ReLU activations.

5. **Visualization**:
   - Gradient magnitudes are plotted across epochs to observe the vanishing and exploding trends.

## Usage

To run the experiment, execute the script:

```bash
Vanishing and Exploding Gradients.ipynb
```

The script will:
1. Load the MNIST dataset.
2. Train neural networks with different architectures.
3. Track gradient magnitudes at each epoch.
4. Plot the gradient magnitudes for each network configuration.

## Results

The results are displayed in a plot showing the average gradient magnitudes across epochs for each configuration. This allows us to observe the impact of depth and activation functions on gradient behavior.

## Observations

- **Sigmoid Activation in Deep Networks**: Gradients tend to vanish in deep networks with Sigmoid activation, as the gradient values approach zero over time.
- **ReLU Activation in Deep Networks**: ReLU can lead to exploding gradients in certain layers of deep networks, causing unstable training.
- **Shallow Networks**: Shallow networks with both Sigmoid and ReLU are less prone to vanishing or exploding gradients, resulting in relatively stable gradient magnitudes.

## Conclusion

This experiment demonstrates that:
- Sigmoid activation in deep networks often leads to vanishing gradients.
- ReLU activation, while mitigating vanishing gradients, can sometimes cause exploding gradients in very deep networks.
- Choosing appropriate activation functions and network depth is essential for stable training of deep neural networks.

Understanding and visualizing these behaviors is crucial for designing effective deep learning models.
