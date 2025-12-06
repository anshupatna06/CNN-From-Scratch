import numpy as np
from conv_forward import conv_forward
from conv_backward import conv_backward
from utils import initialize_filters, initialize_bias

# Dummy input
X = np.random.randn(2, 3, 7, 7)  # (N, C, H, W)
W = initialize_filters(4, 3, 3, 3)
b = initialize_bias(4)

# Forward
out, cache = conv_forward(X, W, b, stride=1, padding=1)
print("Forward Output Shape:", out.shape)

# Dummy gradient (same shape as output)
dout = np.random.randn(*out.shape)

# Backward
dX, dW, db = conv_backward(dout, cache)
print("dX shape:", dX.shape)
print("dW shape:", dW.shape)
print("db shape:", db.shape)

