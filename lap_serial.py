import numpy as np
from sklearn.metrics import mean_squared_error as MSE

# ==========================================================
# Esse arquivo contém uma versão SERIAL do laplaciano que ilustra
# a lógica PARALELA utilziada nos laplacianos da webgpu
# ==========================================================

def lap_serial(img, coef):
    # padding
    pad = len(coef) - 1
    # sizes
    z_sz, x_sz = img.shape[0], img.shape[1]
    Z_sz, X_sz = z_sz + pad * 2, x_sz + pad * 2
    # output array
    out = np.zeros((Z_sz, X_sz), dtype=np.float32)
    # input array with padding
    u = np.zeros((Z_sz, X_sz), dtype=np.float32)
    u[pad:-pad, pad:-pad] = img.astype(np.float32)
    # discrete coefficients
    coef = np.array(coef, dtype=np.float32)

    # for each thread in x direction
    for x in range(x_sz):
        tx = x + pad  # simulate thread x index
        # for each thread in z direction
        for z in range(z_sz):
            tz = z + pad  # simulate thread z index
            lap = 2 * coef[0] * u[tz, tx]  # central position
            for k in range(1, len(coef)):
                # up, down, left, right positions
                lap += coef[k] * (u[tz-k,tx] + u[tz+k,tx] + u[tz,tx-k] + u[tz,tx+k])
            out[tz, tx] = lap

    return out[pad:-pad, pad:-pad]


def lap_pipa(u, c):
    v = (2.0 * c[0] * u).astype(np.float32)
    for k in range(1, len(c)):
        v[k:, :] += c[k] * u[:-k, :]
        v[:-k, :] += c[k] * u[k:, :]
        v[:, k:] += c[k] * u[:, :-k]
        v[:, :-k] += c[k] * u[:, k:]
    return v

# --------------------------
# COEFFICIENTS
# finite difference coefficients
c2 = np.array([-2, 1], dtype=np.float32)
c4 = np.array([-5 / 2, 4 / 3, -1 / 12], dtype=np.float32)
c6 = np.array([-49 / 18, 3 / 2, -3 / 20, 1 / 90], dtype=np.float32)
c8 = np.array([-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560], dtype=np.float32)

# --------------------------
# IMAGE FOR LAPLACIAN
N = 50
# random image
image = np.random.rand(N, N).astype(np.float32)

# --------------------------
# TESTING
im1 = image
im2 = image
# num de iterations
it = 10
for i in range(it):
    v1 = lap_pipa(im1, c8)
    im1 = v1
    v2 = lap_serial(im2, c8)
    im2 = v2
    print(f'It. [{i}]: {MSE(v1, v2)}')