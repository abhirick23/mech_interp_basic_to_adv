# Module 0.1 — Neural Network From Scratch

## What this project is

A small neural network built from scratch in numpy (no PyTorch/TensorFlow for the
actual network), trained on the XOR problem, and then poked at with a very basic
mechanistic interpretability experiment to see if we can figure out what each
individual neuron is doing.

XOR is used as the task because it's the simplest problem that isn't linearly
separable — a single-layer network can't solve it, so it forces the network to
actually use its hidden layer in a meaningful way. That makes it a good toy
example for both learning backprop and for interpretability, since there are
only 3 hidden neurons and 4 possible inputs to reason about.

## The network

- Input: 2 features (the two XOR bits)
- Hidden layer (`dense1`): 2 → 3, followed by ReLU
- Output layer (`dense2`): 3 → 2, followed by Softmax
- Loss: Categorical Cross-Entropy

```
X (2) -> Dense(2,3) -> ReLU -> Dense(3,2) -> Softmax -> Cross-Entropy Loss
```

## Files, in the order they were built

1. **`network.py`**
   Just the forward pass. `Layer_Dense` does `output = inputs @ weights + biases`.
   Two layers are stacked and run on a toy input to check the plumbing works.

2. **`backprop.py`**
   Same forward pass, plus backward methods for every piece:
   - `Layer_Dense.backward` — gradients w.r.t. weights, biases, and inputs
   - `Activation_ReLU.backward` — zeroes the gradient wherever the input was ≤ 0
   - `Activation_Softmax_Loss_CategoricalCrossentropy.backward` — softmax and
     cross-entropy are combined into one class here because their gradients
     simplify beautifully together into `softmax_output - one_hot(y_true)`,
     rather than backpropagating through each separately.

3. **`verify.py`**
   Builds the exact same network in PyTorch, using the *same* weights/biases
   that the numpy version randomly initialized, and runs PyTorch's autograd
   on it. Then compares `dense1.dweights`, `dense1.dbiases`, `dense2.dweights`,
   `dense2.dbiases` from the hand-written backward pass against PyTorch's
   `.grad` values. This is the sanity check that the hand-derived math is
   actually correct, not just "runs without crashing."

4. **`train.py`**
   Adds an actual training loop: forward pass → loss → backward pass → gradient
   descent update (`w -= lr * dw`), repeated for many epochs, with accuracy and
   loss printed periodically. This is what actually solves XOR.

5. **`mech_interp.py`**
   Trains the network the same way, then runs a simple ablation study: after
   training, zero out one hidden neuron's output at a time and re-run the
   forward pass, comparing predictions/loss/accuracy against the untouched
   ("baseline") network. The idea is to see how much each of the 3 hidden
   neurons actually matters, and to use the pattern of failures to guess what
   each neuron might be doing.

## How to run

```
pip install -r requirements.txt
python network.py       # forward pass only
python backprop.py      # forward + backward, loss printed
python verify.py        # gradient check against PyTorch
python train.py         # full training loop on XOR
python mech_interp.py   # train, then ablate each neuron and compare
```

See `conclusion.md` for what the ablation experiment actually showed.
