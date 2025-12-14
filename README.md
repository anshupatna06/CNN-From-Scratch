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
