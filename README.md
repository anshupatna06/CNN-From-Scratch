# CNN-From-Scratch
"DL models implemented from scratch using NumPy and Pandas only"
# 📌 CNN Building Blocks From Scratch (Deep Learning Fundamentals)

A complete low-level implementation of the core operations behind Convolutional Neural Networks — implemented entirely from scratch using NumPy without any DL framework.

This project demonstrates:

✔️ Mathematical understanding

✔️ Low-level architecture fundamentals

✔️ Backpropagation logic

✔️ im2col & col2im vectorization

✔️ How CNNs actually “see” images

✔️ Efficient computation with matrix operations

This repository is your foundation to later build a full CNN from scratch (MNIST, CIFAR-10 etc.).

## ⭐ 1. What is Implemented Here?
### 🔹 1. Convolution Forward Pass (From Scratch)

Using im2col for efficient patch extraction.

Mathematically:
$$\text{out}[n, f, h, w]$$ = $$\sum_{c=0}^{C-1} \sum_{i=0}^{FH-1} \sum_{j=0}^{FW-1}
X[n, c, h+i, w+j] \cdot W[f, c, i, j]$$

### 🔹 2. Convolution Backward Pass (From Scratch)

Computes gradients:

Gradient wrt output: dout

Gradient wrt weights:


$$dW[f, c, i, j]$$ = $$\sum_{n,h,w} X[n,c,h+i,w+j] \cdot dOut[n,f,h,w]$$

Gradient wrt input:


dX = $$\text{col2im}(dX_{col})$$


---

### 🔹 3. im2col Implementation

Transforms patches → columns to convert convolution into matrix multiplication.

Visually:

Image (H×W×C)
  ↓ patches
im2col → matrix (C*FH*FW , H_out * W_out)

This:

Speeds computation

Avoids nested loops

Makes convolution = dot product

### 🔹 4. col2im Implementation

The inverse of im2col — required for backward propagation to reconstruct dX.

## ⭐ 2. Why This Repository Matters

Convolution layers in PyTorch/TensorFlow are black boxes.

This project reveals what’s inside:

✔ How CNNs extract edges, textures, patterns
✔ How filters slide over images
✔ How backprop updates filters
✔ How gradients flow
✔ How patches are vectorized

This level of depth is what engineers working at Microsoft, Google, Meta understand.

This repo demonstrates that you understand CNNs far below the surface.

## ⭐ 3. Folder Structure Explained
📁 src/

Low-level implementations:

File	Description
conv_forward.py	Convolution forward pass
conv_backward.py	Backpropagation for convolution
im2col.py	Convert image → patches
col2im.py	Convert patches → image
utils.py	Utility functions
test_convolution.py	Basic correctness tests
📁 notebooks/

Contains visual demos using Matplotlib.

📁 visuals/

Contains diagrams for readme:

im2col explained

convolution operation diagram

gradient flow

shape transformations

## ⭐ 4. Key Mathematical Formulas
🔹 Output Shape of Convolution
$$H_{out}$$ = $$\frac{H + 2P - FH}{S} + 1$$

$$W_{out}$$ = $$\frac{W + 2P - FW}{S} + 1$$


---

🔹 Convolution as Matrix Multiplication

$$X_{col} \in \mathbb{R}^{(C \cdot FH \cdot FW) \times (H_{out} \cdot W_{out})}$$

$$W_{row} \in \mathbb{R}^{F \times (C \cdot FH \cdot FW)}$$

Out = $$W_{row} \cdot X_{col}$$


---

🔹 Weight Gradient

dW = $$dOut \cdot X_{col}^{T}$$


---

🔹 Input Gradient

dX = $$col2im(W^T \cdot dOut)$$
## ⭐ 5. Running the Demo
pip install -r requirements.txt


Then:

python src/test_convolution.py


Or run Jupyter notebook:

jupyter notebook notebooks/CNN_Convolution_Visual_Demo.ipynb

## ⭐ 6. Roadmap (Coming Next)

You will soon build:

✔ MaxPooling from scratch

✔ ReLU + Softmax

✔ Fully Connected Layer

✔ Mini-Batch Gradient Descent

✔ Adam Optimizer

✔ Full CNN Training on MNIST

✔ Visualizing learned filters

## ⭐ 7. Author

Anshu Pandey
Machine Learning & Deep Learning Learner
Target — Microsoft AI/ML Internship
