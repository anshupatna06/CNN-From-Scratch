import numpy as np

def im2col(X, FH, FW, stride=1, pad=0):
    """
    X shape: (N, C, H, W)
    Returns matrix of shape: (C*FH*FW, N*OH*OW)
    """
    N, C, H, W = X.shape
    H_out = (H + 2*pad - FH)//stride + 1
    W_out = (W + 2*pad - FW)//stride + 1
    
    # Padding
    X_padded = np.pad(
        X,
        pad_width=((0,0),(0,0),(pad,pad),(pad,pad)),
        mode='constant'
    )

    cols = np.zeros((C * FH * FW, N * H_out * W_out))

    col_idx = 0
    for i in range(0, H_out):
        for j in range(0, W_out):
            patch = X_padded[:, :, i*stride:i*stride+FH, j*stride:j*stride+FW]
            cols[:, col_idx:col_idx+N] = patch.reshape(N, -1).T
            col_idx += N

    return cols, H_out, W_out

