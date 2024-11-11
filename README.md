Here’s an enhanced README to include each experiment in separate folders with detailed instructions:

---

# CS-704 Deep Reinforcement Learning Lab

Welcome to the **CS-704 Deep Reinforcement Learning Lab** repository! This repository contains code and resources for the lab experiments covered in the CS-704 course, aimed at exploring key concepts in deep and reinforcement learning, including neural networks, optimization techniques, and dimensionality reduction.

## Table of Contents

- [Experiments Overview](#experiments-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Experiments Overview

Each experiment in this lab is organized into a separate folder, containing a Jupyter Notebook (`.ipynb`) with code and explanations for the respective topic.

| Sr.No | Experiment Title | Folder Name |
|-------|-------------------|-------------|
| 01    | MLP Models | `MLP Models` |
| 02    | Numpy  | `Numpy` |
| 03    | Implementing McCulloch-Pitts Neuron | `McCulloch_Pitts_Neuron` |
| 04    | Activation Functions | `Activation_Functions` |
| 05    | Gradient Descent Variants | `Gradient_Descent_Variants` |
| 06    | Advanced Optimization Algorithms | `Optimization_Algorithms` |
| 07    | Eigenvalue Decomposition | `Eigenvalue_Decomposition` |
| 08    | Vanishing and Exploding Gradients | `Vanishing_Exploding_Gradients` |
| 09    | Autoencoders and PCA | `Autoencoders_PCA` |
| 10    | Regularization Techniques | `Regularization_Techniques` |
| 11    | Denoising Autoencoder | `Denoising_Autoencoder` |
| 12    | Batch Normalization vs. Instance Normalization | `Batch_vs_Instance_Normalization` |

---

## Installation

To get started with this repository, follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tanvisharma31/CS-704_Deep-Reinforcement_Learning_Lab.git
   cd CS-704_Deep-Reinforcement_Learning_Lab
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure `requirements.txt` includes necessary packages such as `numpy`, `tensorflow`, `keras`, `matplotlib`, and others.

---

## Usage

### Running Experiments

Each experiment folder contains a Jupyter Notebook with a complete implementation of the topic. Follow the steps below to run each notebook:

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Navigate to the Experiment Folder**:
   In the Jupyter Notebook interface, navigate to the folder of the desired experiment, and open the corresponding `.ipynb` file.

3. **Run the Code Cells**:
   Each notebook is structured with explanations and code cells. Run each cell sequentially to observe the outputs and experiment with the code.

---

## Experiment Details

### 01. Implementing McCulloch-Pitts Neuron
- **Objective**: Simulate the behavior of a McCulloch-Pitts neuron and observe responses to various inputs.

### 02. Activation Functions
- **Objective**: Compare different activation functions (ReLU, Sigmoid, Tanh) in a simple neural network.

### 03. Gradient Descent Variants
- **Objective**: Implement and compare Gradient Descent, Momentum-based GD, Nesterov Accelerated GD, and Stochastic GD on a linear regression problem.

### 04. Advanced Optimization Algorithms
- **Objective**: Experiment with AdaGrad, RMSProp, and Adam optimizers in training a neural network.

### 05. Eigenvalue Decomposition
- **Objective**: Perform eigenvalue decomposition on a sample matrix and visualize the results.

### 06. Vanishing and Exploding Gradients
- **Objective**: Train neural networks with different architectures to observe the effects of vanishing and exploding gradients.

### 07. Autoencoders and PCA
- **Objective**: Implement a basic autoencoder and compare its performance to Principal Component Analysis (PCA) for dimensionality reduction.

### 08. Regularization Techniques
- **Objective**: Explore the effects of L2 regularization, Dropout, and noise injection on training an autoencoder.

### 09. Denoising Autoencoder
- **Objective**: Implement a denoising autoencoder and evaluate its effectiveness in reconstructing noisy images.

### 10. Batch Normalization vs. Instance Normalization
- **Objective**: Compare model performance using Batch Normalization and Instance Normalization.

---

## Contributing

We welcome contributions to improve this repository. To contribute, please follow these steps:

1. **Fork the repository**.
2. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes** and commit them:
   ```bash
   git commit -m "Add your message here"
   ```
4. **Push to your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Create a Pull Request** detailing your changes.

---

## Contact

For questions or issues, contact Tanvi Sharma at [2004tanvisharma@gmail.com](mailto:2004tanvisharma@gmail.com).

---

Thank you for exploring the CS-704 Deep Reinforcement Learning Lab repository! Enjoy your learning journey! 🚀