import math
from numba import cuda as nbcuda
import numpy as np
from sklearn.metrics import mean_squared_error as MSE

# Esse arquivo contém a versão do laplaciano implementado em numba cuda com padding e sua tradução para a versão
# "serial".
# OBS: É necessário ter as depêndencias do CUDA instaladas para executar esse script.

def lap_serial(img, coef, it):
    # padding and field size
    num_coef = len(coef)
    pad = num_coef - 1
    x, z = img.shape[0], img.shape[1]
    X, Z = x + pad * 2, z + pad * 2
    # create field with padding and fulfill center
    data = np.zeros((X, Z), dtype=np.float32)
    lap = np.zeros((X, Z), dtype=np.float32)
    data[pad:-pad, pad:-pad] = img.astype(np.float32)
    coef = np.float32(coef)
    # define number of blocks and threads
    threadsperblock = z
    blockspergrid = x

    for b in range(blockspergrid):
        j = b + pad
        for t in range(threadsperblock):
            i = t + pad
            acc = 0.0
            for k in range(1, num_coef):
                acc += coef[k] * data[i - k, j]
                acc += coef[k] * data[i + k, j]
                acc += coef[k] * data[i, j - k]
                acc += coef[k] * data[i, j + k]
            lap[i, j] = acc
    return lap[pad:-pad, pad:-pad]




def lap_nbcuda_pad(img, coef, it):
    # padding and field size
    pad = len(coef) - 1
    x, z = img.shape[0], img.shape[1]
    X, Z = x + pad * 2, z + pad * 2
    # create field with padding and fulfill center
    data = np.zeros((X, Z), dtype=np.float32)
    data[pad:-pad, pad:-pad] = img.astype(np.float32)
    coef = np.float32(coef)
    # send data to gpu
    lap = nbcuda.to_device(np.ascontiguousarray(data, dtype=np.float32))
    # lap = nbcuda.device_array((X, Z), dtype=np.float32)
    coef_gpu = nbcuda.to_device(coef)
    # define number of blocks and threads
    threadsperblock = z
    blockspergrid = x
    # call numba cuda kernel
    lap_kernel_pad[blockspergrid, threadsperblock](lap, coef_gpu, np.int32(it))
    # return gpu data
    return lap.copy_to_host()[pad:-pad, pad:-pad]


@nbcuda.jit("void(float32[:, :], float32[:], int32)")
def lap_kernel_pad(lap, coef, it):
    num_coef = len(coef)
    pad = num_coef - 1
    x = nbcuda.blockIdx.x + pad
    z = nbcuda.threadIdx.x + pad

    for i in range(it):
        acc = 2.0 * coef[0] * lap[x, z]
        for k in range(1, num_coef):
            acc += coef[k] * lap[x - k, z]
            acc += coef[k] * lap[x + k, z]
            acc += coef[k] * lap[x, z - k]
            acc += coef[k] * lap[x, z + k]
        nbcuda.syncthreads()
        lap[x, z] = acc


def lap_pipa(u, c):
    v = 2.0 * c[0] * u
    for k in range(1, len(c)):
        v[k:, :] += c[k] * u[:-k, :]
        v[:-k, :] += c[k] * u[k:, :]
        v[:, k:] += c[k] * u[:, :-k]
        v[:, :-k] += c[k] * u[:, k:]
    return v

