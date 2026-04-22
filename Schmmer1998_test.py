import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from framework import file_m2k, file_civa

D = 24.2          # Espessura (mm)
fs = 125e6        # Taxa de amostragem (Hz)
epsilon = 0.05  # Valor para evitar divisão por zero

FRONT_START, FRONT_END = 0, 500
BACK_START, BACK_END = 750, 1250

f_min, f_max = 2.5, 7.5  # MHz


ensaio_real = file_m2k.read("./Mono_5MHz_ferro.m2k", 5, 0.5, 'Gaussian')
signal = ensaio_real.ascan_data[:,0,0,0]

front = signal[FRONT_START:FRONT_END]
back = signal[BACK_START:BACK_END]

n_fft = int(2 ** np.ceil(np.log2(max(len(front), len(back)) * 4)))

F_mag = np.abs(fft(front, n_fft))
B_mag = np.abs(fft(back, n_fft))

epsilon = (epsilon) * np.max(F_mag)

calc = (B_mag * F_mag) / (F_mag**2 + epsilon**2)
calc = np.clip(calc, 1e-20, None)

alpha = -np.log(calc) / (2.0 * D)

freq = fftfreq(n_fft, 1.0/fs)
pos_mask = freq >= 0
freq_MHz = freq[pos_mask] / 1e6
alpha_pos = alpha[pos_mask]

plt.figure(0)

mask = (freq_MHz >= f_min) & (freq_MHz <= f_max)
freq_band = freq_MHz[mask]
alpha_band = alpha_pos[mask]


valid = np.isfinite(alpha_band) & (alpha_band > 0) & (alpha_band < 0.1)
freq_valid = freq_band[valid]
alpha_valid = alpha_band[valid]

plt.plot(freq_valid, alpha_valid, 'b-', linewidth=1.5)

plt.xlabel('Frequency (MHz)', fontsize=12)
plt.ylabel('Np/mm', fontsize=12)
plt.show()