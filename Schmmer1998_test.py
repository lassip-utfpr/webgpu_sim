import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1
from framework import file_m2k

# --- parametros ---
D      = 24.2e-3       # espessura (m)
a      = (0.25 * 25.4e-3) / 2   # raio transdutor (m)
cp     = 5940.0        # velocidade P (m/s)
fs     = 125e6
eps_f  = 0.05
f_min, f_max = 1.75, 5.25   # MHz

# gates em passos — eco 1 e eco 2 de fundo
ECO1 = (900, 1250)
ECO2 = (1950, 2300)

# --- carregar ---
ensaio = file_m2k.read("./Mono_3.5MHz_ferro.m2k", 3.5, 0.5, 'Gaussian')
signal = ensaio.ascan_data[:, 0, 0, 0].astype(np.float64)
t      = np.arange(len(signal)) / fs

# --- extrair ecos com janela Hann ---
def extrair(t, sig, sa, sb):
    mask = (np.arange(len(sig)) >= sa) & (np.arange(len(sig)) <= sb)
    seg  = sig[mask]
    out  = np.zeros_like(sig)
    out[mask] = seg * np.hanning(len(seg))
    return out

e1 = extrair(t, signal, *ECO1)
e2 = extrair(t, signal, *ECO2)

NFFT  = int(2 ** np.ceil(np.log2(len(signal) * 2)))
V1    = np.fft.rfft(e1, n=NFFT)
V2    = np.fft.rfft(e2, n=NFFT)
freqs = np.fft.rfftfreq(NFFT, d=1.0/fs)

# --- correcao de difracao Eq. 9.56 ---
def Dp(freqs, a, D_eff, cp):
    f   = np.where(freqs == 0, 1e-10, freqs)
    arg = 2*np.pi*f/cp * a**2 / (2*D_eff)
    return 1.0 - np.exp(1j*arg) * (j0(arg) - 1j*j1(arg))

dp1 = Dp(freqs, a, D,     cp)
dp2 = Dp(freqs, a, 2.0*D, cp)

# --- filtro de Wiener Eq. 9.60 ---
F   = V1 * (np.abs(dp2) / (np.abs(dp1) + 1e-30))
B   = V2
aF  = np.abs(F);  aB = np.abs(B)
eps = eps_f * aF.max()

e2aD  = (aB * aF) / (aF**2 + eps**2)
alpha = -np.log(np.clip(e2aD, 1e-30, None)) / (2.0 * D)   # Np/m

# --- plot na banda ---
fMHz = freqs / 1e6
mask = (fMHz >= f_min) & (fMHz <= f_max)

plt.figure()
plt.plot(fMHz[mask], alpha[mask] / 1000.0, 'k-', lw=1.5)   # Np/m -> Np/mm
plt.xlabel('Frequencia (MHz)')
plt.ylabel('alpha (Np/mm)')
plt.grid(True, alpha=0.3)
plt.show()