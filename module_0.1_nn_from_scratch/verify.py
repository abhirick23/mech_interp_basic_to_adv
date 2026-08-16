import numpy as np
import torch

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
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


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


# ---- numpy forward + backward ----
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 2)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print("Numpy loss:", loss)

# ---- identical network in PyTorch, same weights/biases/data ----
X_t = torch.tensor(X, dtype=torch.float64)
y_t = torch.tensor(y, dtype=torch.long)

w1 = torch.tensor(dense1.weights, dtype=torch.float64, requires_grad=True)
b1 = torch.tensor(dense1.biases, dtype=torch.float64, requires_grad=True)
w2 = torch.tensor(dense2.weights, dtype=torch.float64, requires_grad=True)
b2 = torch.tensor(dense2.biases, dtype=torch.float64, requires_grad=True)

z1 = X_t @ w1 + b1
a1 = torch.relu(z1)
z2 = a1 @ w2 + b2

torch_loss = torch.nn.functional.cross_entropy(z2, y_t)
torch_loss.backward()

print("Torch loss:", torch_loss.item())

# ---- compare gradients ----
checks = [
    ("dense1.dweights", dense1.dweights, w1.grad.numpy()),
    ("dense1.dbiases", dense1.dbiases, b1.grad.numpy()),
    ("dense2.dweights", dense2.dweights, w2.grad.numpy()),
    ("dense2.dbiases", dense2.dbiases, b2.grad.numpy()),
]

all_match = True
for name, np_grad, torch_grad in checks:
    match = np.allclose(np_grad, torch_grad, atol=1e-6)
    all_match &= match
    print(f"{name}: match={match}")
    print("  numpy:\n", np.round(np_grad, 6))
    print("  torch:\n", np.round(torch_grad, 6))
    assert match, f"{name} does not match PyTorch autograd gradient"

print("\nAll gradients match PyTorch autograd to 6 decimal places:", all_match)

