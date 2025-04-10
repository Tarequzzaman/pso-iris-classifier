# PSO-Iris-Classifier 🧠🐦

This project implements and compares two neural network training approaches — traditional backpropagation and Particle Swarm Optimization (PSO) — for a binary classification task using the Iris dataset (Setosa vs Versicolor).

## 🔍 Overview

Hyperparameter and weight optimization are important part of  neural network performance. This project explores an evolutionary computation approach (PSO) to train a neural network without using gradient descent, comparing it against a standard backpropagation-based neural network.

### Features
- Binary classification using the Iris dataset
- Fully connected feedforward neural network
- PSO-based training using informants and swarm intelligence
- Performance evaluation with accuracy and cross-entropy loss
- Statistical analysis using paired t-test

## 🧪 Libraries used 
- Python 3.11
- NumPy
- Scikit-learn
- TensorFlow (for traditional NN)
- SciPy (for t-test)
- Docker & Docker Compose (for containerized execution)

## 📊 Model Architectures

- **Traditional NN**:
  - Input: 4 features
  - Hidden: 5 neurons, sigmoid
  - Output: 1 neuron, sigmoid
  - Optimizer: Adam, Loss: Binary Crossentropy

- **PSO-NN**:
  - Encodes all weights and biases in a 31D particle
  - Informant-based swarm structure
  - Loss used as fitness function

## 🚀 How to Run

### 📦 Local Setup
```bash
pip install -r requirements.txt
python pso_iris.py
```

### Docker setup 

#### 🛠 Build and Run the Project

```bash
docker-compose up --build
```

#### Stop docker container
```bash
docker-compose down
```
