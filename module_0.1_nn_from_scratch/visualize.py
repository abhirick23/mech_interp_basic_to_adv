import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([0, 1, 1, 0])


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct_confidences)


class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


# ---- train ----
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 2)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

learning_rate = 1.0
epochs = 10000

for epoch in range(epochs):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_activation.forward(dense2.output, y)

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    dense1.weights -= learning_rate * dense1.dweights
    dense1.biases -= learning_rate * dense1.dbiases
    dense2.weights -= learning_rate * dense2.dweights
    dense2.biases -= learning_rate * dense2.dbiases

print("Trained. dense1.weights:\n", dense1.weights)


def predict(inputs, kill_neuron=None):
    """Full forward pass, optionally zeroing one hidden neuron's activation."""
    dense1.forward(inputs)
    activation1.forward(dense1.output)
    a1 = activation1.output.copy()
    if kill_neuron is not None:
        a1[:, kill_neuron] = 0
    dense2.forward(a1)
    loss_activation.activation.forward(dense2.output)
    return loss_activation.activation.output  # softmax probabilities


# ---- build a grid over the input space ----
grid_res = 200
xx, yy = np.meshgrid(
    np.linspace(-0.5, 1.5, grid_res),
    np.linspace(-0.5, 1.5, grid_res),
)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# --- row 1: decision boundary + each hidden neuron's activation ---
probs = predict(grid_points)
class1_prob = probs[:, 1].reshape(xx.shape)

ax = axes[0, 0]
cs = ax.contourf(xx, yy, class1_prob, levels=50, cmap="RdBu_r", vmin=0, vmax=1)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="black", s=150, linewidths=1.5)
ax.set_title("Decision boundary (P(class=1))")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
fig.colorbar(cs, ax=ax)

dense1.forward(grid_points)
activation1.forward(dense1.output)
hidden_acts = activation1.output  # (N, 3)

for i in range(3):
    ax = axes[1, i]
    act_grid = hidden_acts[:, i].reshape(xx.shape)
    cs = ax.contourf(xx, yy, act_grid, levels=50, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="black", s=150, linewidths=1.5)
    ax.set_title(f"Hidden neuron {i} activation (ReLU output)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.colorbar(cs, ax=ax)

# --- top-right two panels: decision boundary with each neuron ablated ---
for col, kill in zip([1, 2], [1, 2]):
    probs_ablated = predict(grid_points, kill_neuron=kill)
    class1_prob_ablated = probs_ablated[:, 1].reshape(xx.shape)
    ax = axes[0, col]
    cs = ax.contourf(xx, yy, class1_prob_ablated, levels=50, cmap="RdBu_r", vmin=0, vmax=1)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="black", s=150, linewidths=1.5)
    ax.set_title(f"Decision boundary, neuron {kill} ablated")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.colorbar(cs, ax=ax)

plt.tight_layout()
plt.savefig("xor_visualization.png", dpi=150)
print("Saved xor_visualization.png")
plt.show()
