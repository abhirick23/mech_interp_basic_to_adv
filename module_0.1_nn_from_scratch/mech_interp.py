import numpy as np

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


dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 2)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

learning_rate = 1.0
epochs = 10000
print_every = 1000

# ---- train normally, no ablation ----
for epoch in range(epochs):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    if epoch % print_every == 0:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    dense1.weights -= learning_rate * dense1.dweights
    dense1.biases -= learning_rate * dense1.dbiases
    dense2.weights -= learning_rate * dense2.dweights
    dense2.biases -= learning_rate * dense2.dbiases

# print("\nFinal weights and biases:")
# print("Dense layer 1 weights:\n", dense1.weights)
# print("Dense layer 2 weights:\n", dense2.weights)

# ---- post-training ablation: kill neuron 0 of dense1 and compare ----
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
baseline_loss = loss_activation.forward(dense2.output, y)
baseline_preds = np.argmax(loss_activation.output, axis=1)
baseline_acc = np.mean(baseline_preds == y)

for neuron_idx in range(3):
    patched = dense1.output.copy()
    patched[:, neuron_idx] = 0  # kill neuron
    activation1.forward(patched)
    dense2.forward(activation1.output)
    patched_loss = loss_activation.forward(dense2.output, y)
    patched_preds = np.argmax(loss_activation.output, axis=1)
    patched_acc = np.mean(patched_preds == y)

    print(f"\nAblation: dense1 neuron {neuron_idx} zeroed")
    print(f"baseline -> loss: {baseline_loss:.4f}, acc: {baseline_acc:.3f}, preds: {baseline_preds}")
    print(f"ablated  -> loss: {patched_loss:.4f}, acc: {patched_acc:.3f}, preds: {patched_preds}")