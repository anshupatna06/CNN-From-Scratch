def col2im(cols, X_shape, FH, FW, stride=1, pad=0):
    N, C, H, W = X_shape
    H_out = (H + 2*pad - FH)//stride + 1
    W_out = (W + 2*pad - FW)//stride + 1

    X_padded = np.zeros((N, C, H + 2*pad, W + 2*pad))
    col_idx = 0

    for i in range(0, H_out):
        for j in range(0, W_out):
            patch = cols[:, col_idx:col_idx+N].T.reshape(N, C, FH, FW)
            X_padded[:, :, i*stride:i*stride+FH, j*stride:j*stride+FW] += patch
            col_idx += N

    return X_padded[:, :, pad:H+pad, pad:W+pad]
