# Conclusion — Ablation Results on the Trained XOR Network

## Setup

After training to 100% accuracy on XOR (loss ~0), `dense1`'s learned weights
were:

```
neuron 0: [ 1.731,  1.731]
neuron 1: [-2.569,  3.254]
neuron 2: [ 3.122, -3.122]
```

(each row is `[weight_from_input_0, weight_from_input_1]` for that neuron)

## Step 1 — look at the weights before touching anything

- **Neuron 0** has two almost-identical, same-sign weights. That means it
  reacts almost the same way regardless of *which* input bit is 1 — it mostly
  just responds to "how many bits are on", not "which one." Hypothesis: this
  neuron is somewhat redundant / not doing the important XOR-specific work.
- **Neurons 1 and 2** have opposite-sign weight pairs, and are roughly mirror
  images of each other (`[-2.57, 3.25]` vs `[3.12, -3.12]`). That pattern looks
  like each one is specialized to fire for one specific "only one bit is on"
  case — i.e. one of the two inputs that should output 1 — and the other input
  suppresses it.

## Step 2 — test the hypothesis by ablating each neuron

Ablation = force one neuron's output to 0 after training, then re-run the
forward pass on all 4 XOR inputs and see what breaks.

Baseline (nothing ablated): predictions `[0, 1, 1, 0]`, accuracy 1.000, loss ≈ 0.0001

| Neuron zeroed | Predictions   | Accuracy | Loss   |
|---------------|---------------|----------|--------|
| 0             | `[0, 1, 1, 0]`| 1.000    | 0.0033 |
| 1             | `[0, 0, 1, 0]`| 0.750    | 2.0836 |
| 2             | `[0, 1, 0, 0]`| 0.750    | 2.0836 |

## Step 3 — interpretation

- **Killing neuron 0 changes nothing** (predictions identical, loss barely
  moves). This confirms the hypothesis: neuron 0 isn't load-bearing for the
  actual XOR decision. The network apparently found a solution where the
  "which bit is on" work is entirely handled by neurons 1 and 2, and neuron 0
  is redundant capacity.
- **Killing neuron 1 breaks input `(0,1)`**, flipping its prediction from 1 to
  0. That's exactly the one input neuron 1's weights are shaped to detect
  (input 1 firing, input 0 suppressing).
- **Killing neuron 2 breaks input `(1,0)`** instead — the mirror-image input,
  which matches its mirror-image weights.

So the network didn't just "solve XOR" as an opaque black box — it split the
work cleanly: neuron 1 is a detector for one of the two positive cases,
neuron 2 is a detector for the other positive case, and neuron 0 is extra
capacity the network didn't end up needing. Each of the two "should be 1"
inputs has exactly one neuron responsible for it, and losing that neuron loses
exactly that one input — nothing else.

## Takeaway

This is a working example of the basic mech-interp loop on the smallest
possible network:

1. form a hypothesis from the raw weights,
2. test it with a targeted intervention (ablation),
3. check whether the failure pattern matches the hypothesis.

Here, it did — the hypothesis is accepted for this trained network. (Note:
this is specific to *this* random seed / training run. A different
initialization could easily split the work across neurons differently.)
