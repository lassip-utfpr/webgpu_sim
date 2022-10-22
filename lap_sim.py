from sklearn.metrics import mean_squared_error as MSE
import numpy as np
import matplotlib.pyplot as plt
from lap_webgpu import lap_for, lap_pad, lap_pipa
from time import time
from sklearn.metrics import mean_squared_error as MSE
import wgpu.backends.rs  # Select backend
from wgpu.utils import compute_with_buffers  # Convenience function

# ==========================================================
# Esse arquivo contém as simulações onde somente o laplaciano é calculado na GPU.
# ==========================================================

# field config
Nz = 256
Nx = 256
Nt = 2048
dtype = 'float32'
c = .2
# source term
t = np.arange(Nt)
s = np.exp(-(t - Nt / 10) ** 2 / 500)
s[:-1] -= s[1:]
# plt.plot(t, s)


# COEFFICIENTS
# finite difference coefficient
c2 = np.array([-2, 1], dtype=dtype)
c4 = np.array([-5 / 2, 4 / 3, -1 / 12], dtype=dtype)
c6 = np.array([-49 / 18, 3 / 2, -3 / 20, 1 / 90], dtype=dtype)
c8 = np.array([-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560], dtype=dtype)

# Simulação onde SOMENTE o laplaciano é calculado na GPU
def sim_lap_pad():
    u = np.zeros((Nz, Nx, Nt), dtype=dtype)
    for k in range(2, Nt):
        u[:, :, k] = -u[:, :, k - 2] + 2 * u[:, :, k - 1] + c ** 2 * lap_pad(u[:, :, k - 1], c8, 1)
        u[Nz // 2, Nx // 2, k] += s[k]
    return u

# Simulação onde SOMENTE o laplaciano é calculado na GPU
def sim_lap_for():
    u = np.zeros((Nz, Nx, Nt), dtype=dtype)

    for k in range(2, Nt):
        u[:, :, k] = -u[:, :, k - 2] + 2 * u[:, :, k - 1] + c ** 2 * lap_for(u[:, :, k - 1], c8, 1)
        u[Nz // 2, Nx // 2, k] += s[k]
    return u

# Simulação serial
def sim_full():
    u = np.zeros((Nz, Nx, Nt), dtype=dtype)

    for k in range(2, Nt):
        u[:, :, k] = -u[:, :, k - 2] + 2 * u[:, :, k - 1] + c ** 2 * lap_pipa(u[:, :, k - 1], c8)
        u[Nz // 2, Nx // 2, k] += s[k]
    return u


t_pad = time()
u_pad = sim_lap_pad()
t_pad = time() - t_pad

t_for = time()
u_for = sim_lap_for()
t_for = time() - t_for

t_ser = time()
u_ser = sim_full()
t_ser = time() - t_ser

print(f'MSE entre as duas simulações: {MSE(u_pad[:, :, -1], u_for[:, :, -1])}')
print(f'MSE entre Pad e Ref: {MSE(u_pad[:, :, -1], u_ser[:, :, -1])}')
print(f'MSE entre For e Ref: {MSE(u_for[:, :, -1], u_ser[:, :, -1])}')
print(f'TEMPO - {Nt} pontos de tempo:\nPadding: {t_pad:.3}s\nFor: {t_for:.3}s\nSerial: {t_ser:.3}s')

plt.figure(1)
plt.title('Sim. com lap pad na GPU')
plt.imshow(u_pad[:, :, -1], aspect='auto')
plt.figure(2)
plt.title('Sim. com lap for na GPU')
plt.imshow(u_for[:, :, -1], aspect='auto')
plt.figure(3)
plt.title('Sim. serial')
plt.imshow(u_ser[:, :, -1], aspect='auto')
plt.show()