# Neural Network from Scratch (XOR)

This is a beginner-friendly neural network built from scratch using **NumPy** to solve the classic **XOR** problem.

It’s part of my journey into AI/ML — learning the fundamentals by building things from the ground up.

---

## What It Does

- Takes 2 inputs (like `[0, 1]`, `[1, 0]`, etc.)
- Predicts the XOR output (`1` if inputs are different, `0` if same)
- Uses:
  - Sigmoid activation
  - NumPy for matrix operations
  - Binary cross-entropy loss

---

## Project Structure

- `inputs` – XOR input combinations
- `weights` & `bias` – randomly initialized
- `sigmoid()` – squashes outputs between 0 and 1
- `predict()` – forward pass of the network
- `loss()` – measures prediction error using binary cross-entropy

---

## Why XOR?

The XOR problem is a classic ML test that simple linear models **can’t solve** — it requires **non-linear decision boundaries**, which makes it perfect for learning why neural networks matter.

---

## What I’m Learning

- NumPy fundamentals
- Neural network logic (forward pass, activation)
- Binary classification
- Loss functions like binary cross-entropy
- How to build AI from scratch, not just use libraries

---

## Next Goals

- Add training (gradient descent + backpropagation)
- Visualize loss over time
- Expand to more complex logic gates or datasets
- Learn and implement more layers

---

## Run It Yourself

```bash
# Requirements
pip install numpy

# Run
python main.py
