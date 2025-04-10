# =======================
# 1. Problem Setup + Data Preprocessing
# =======================
import numpy as np  # For numerical operations
from sklearn.datasets import load_iris  # Load Iris dataset
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import StandardScaler  # For feature normalization
from scipy.stats import ttest_rel  # For statistical comparison
import tensorflow as tf  # For building the baseline neural network
from tensorflow.keras import layers, Sequential  # For defining NN layers

# Load Iris dataset
iris = load_iris()
X = iris.data  # Feature matrix
y = (iris.target == 0).astype(int)  # Convert to binary: 1 if Setosa, else 0
X = X[y != 2]  # Remove class 2 (only Setosa and Versicolor)
y = y[y != 2]  # Update target accordingly

# Normalize features using standard scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =======================
# 2. Traditional NN (Backpropagation)
# =======================
# Build a simple feedforward NN with 1 hidden layer (5 neurons)
model = Sequential([
    layers.Input(shape=(4,)),  # Define input layer explicitly
    layers.Dense(5, activation='sigmoid'),  # Hidden layer
    layers.Dense(1, activation='sigmoid')   # Output layer
])


# Compile the model with optimizer, loss and metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on training data
model.fit(X_train, y_train, epochs=50, verbose=0)

# Evaluate the model on test data
loss_bp, acc_bp = model.evaluate(X_test, y_test, verbose=0)
print(f"[Backprop NN] Accuracy: {acc_bp*100:.2f}%, Loss: {loss_bp:.4f}")

# =======================
# 3. PSO-Optimized NN
# =======================
# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Binary cross-entropy loss function
def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-5, 1 - 1e-5)  # Clip predictions to avoid log(0)
    return np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

class Particle:
    def __init__(self, dim):
        self.position = np.random.uniform(-1, 1, dim)
        self.velocity = np.random.uniform(-0.1, 0.1, dim)
        self.best_position = self.position.copy()
        self.best_fitness = np.inf
        self.swarm = None  # Will be set later
        self.informants = []  # Will be set later

    def set_swarm_and_informants(self, swarm):
        self.swarm = swarm
        self.informants = np.random.choice(swarm, size=3, replace=False)

    def get_informant_best(self):
        return min(self.informants, key=lambda p: p.best_fitness).best_position

    def feedforward(self, X):
        idx = 0
        w1 = self.position[idx:idx+20].reshape((4, 5)); idx += 20
        b1 = self.position[idx:idx+5]; idx += 5
        w2 = self.position[idx:idx+5].reshape((5, 1)); idx += 5
        b2 = self.position[idx:]
        h = sigmoid(np.dot(X, w1) + b1)
        return sigmoid(np.dot(h, w2) + b2)

    def evaluate(self, X, y):
        pred = self.feedforward(X).flatten()
        loss = binary_cross_entropy(y, pred)
        if loss < self.best_fitness:
            self.best_fitness = loss
            self.best_position = self.position.copy()
        return loss


# Define PSO parameters
dim = 31  # Total number of weights and biases
swarm_size = 15
iterations = 50
w, c1, c2 = 0.7, 1.5, 1.5  # Inertia and cognitive/social coefficients

# Step 1: Create all particles without swarm references
swarm = [Particle(dim) for _ in range(swarm_size)]

# Step 2: Assign the swarm reference and informants
for particle in swarm:
    particle.set_swarm_and_informants(swarm)


# Initialize global best variables
global_best_pos = None
global_best_fit = np.inf

# Run PSO loop
for it in range(iterations):
    for p in swarm:
        fit = p.evaluate(X_train, y_train)  # Evaluate particle
        if fit < global_best_fit:  # Update global best
            global_best_fit = fit
            global_best_pos = p.position.copy()

    for p in swarm:
        informant_best = p.get_informant_best()  # Get best informant position
        r1, r2 = np.random.rand(dim), np.random.rand(dim)  # Random vectors
        cognitive = c1 * r1 * (p.best_position - p.position)  # Cognitive component
        social = c2 * r2 * (informant_best - p.position)  # Social component
        p.velocity = w * p.velocity + cognitive + social  # Update velocity
        p.position += p.velocity  # Update position

    print(f"[PSO] Iter {it+1}/{iterations}, Best Loss: {global_best_fit:.4f}")

# Evaluate final PSO solution
# Evaluate final PSO model using the best-found weights
final_pso = Particle(dim)
final_pso.set_swarm_and_informants(swarm)
final_pso.position = global_best_pos  # Set best weights

pso_preds = final_pso.feedforward(X_test).flatten()
pso_labels = (pso_preds > 0.5).astype(int)
acc_pso = np.mean(pso_labels == y_test)
final_loss = binary_cross_entropy(y_test, pso_preds)

print(f"[PSO NN] Accuracy: {acc_pso*100:.2f}%, Loss: {final_loss:.4f}")


# =======================
# 4. Performance Comparison
# =======================
bp_preds = model.predict(X_test).flatten()  # Predictions from backprop NN
t_stat, p_val = ttest_rel(bp_preds, pso_preds)  # Perform paired t-test
print(f"[T-Test] t={t_stat:.4f}, p={p_val:.4f}")  # Display test result