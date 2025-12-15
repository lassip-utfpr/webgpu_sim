import numpy as np
import matplotlib.pyplot as plt
from framework import file_m2k
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

data1 = file_m2k.read(r"C:\Users\Henrique\PycharmProjects\webgpu_sim\H_facin_teste1.m2k", 5, 0.5, 'Gaussian') #Se for identificar no teste 1, os tempos são de 450
data2 = file_m2k.read(r"C:\Users\Henrique\PycharmProjects\webgpu_sim\H_facin_teste2.m2k", 5, 0.5, 'Gaussian') #Se for identificar no teste 2, os tempos são de 700
result_real2 = data2.ascan_data[:, 0, :, 0]

sinal_emitido = result_real2[:451,4]
sinal_recebido = result_real2[3000:3451,4]

print("Identificação do sistema no domínio da frequência")
print("Método: H(f) = Y(f) / X(f) com filtro passa-banda")
print(f"Sinal emitido  - Max: {np.max(np.abs(sinal_emitido)):.3f}")
print(f"Sinal recebido - Max: {np.max(np.abs(sinal_recebido)):.3f}\n")

# Parametros
fs = 100e6
n = len(sinal_recebido)
time = np.arange(n) / fs * 1e6
freq = fftfreq(n, 1 / fs) / 1e6

# FFT dos sinais
X = fft(sinal_emitido)
Y = fft(sinal_recebido)

# Identificação: H(f) = Y(f) / X(f)
H = np.zeros_like(X, dtype=complex)
epsilon = 1e-10  # Evitar divisão por zero

for i in range(len(X)):
    if np.abs(X[i]) > epsilon:
        H[i] = Y[i] / X[i]
    else:
        H[i] = 0

# Filtro passa-banda em H(f)
fmin = 1.5e6  # MHz
fmax = 4.5e6  # MHz
freq_array = fftfreq(len(H), 1 / fs)
bandpass_mask = (np.abs(freq_array) >= fmin) & (np.abs(freq_array) <= fmax)
H_filtered = H * bandpass_mask

# Resposta impulsiva por IFFT
h = np.real(ifft(H_filtered))

# Verificação: reconvolução
sinal_reconstruido_full = np.convolve(sinal_emitido, h, mode='full')

# Detectar delay
correlacao_cruzada = np.correlate(sinal_recebido, sinal_reconstruido_full, mode='full')
delay_samples = np.argmax(correlacao_cruzada) - len(sinal_reconstruido_full) + 1

# Alinhar sinal reconstruido
if delay_samples < 0:
    start_idx = abs(delay_samples)
    end_idx = start_idx + len(sinal_recebido)
    if end_idx <= len(sinal_reconstruido_full):
        sinal_reconstruido = sinal_reconstruido_full[start_idx:end_idx]
    else:
        sinal_reconstruido = np.zeros_like(sinal_recebido)
else:
    if delay_samples + len(sinal_recebido) <= len(sinal_reconstruido_full):
        sinal_reconstruido = sinal_reconstruido_full[delay_samples:delay_samples + len(sinal_recebido)]
    else:
        sinal_reconstruido = np.zeros_like(sinal_recebido)

# Metricas
erro = sinal_recebido - sinal_reconstruido
erro_rms = np.sqrt(np.mean(erro ** 2))
sinal_rms = np.sqrt(np.mean(sinal_recebido ** 2))
erro_percentual = (erro_rms / sinal_rms) * 100 if sinal_rms > 0 else 0

if np.std(sinal_recebido) > 0 and np.std(sinal_reconstruido) > 0:
    correlacao = np.corrcoef(sinal_recebido, sinal_reconstruido)[0, 1]
else:
    correlacao = 0.0

print(f"Resultados:")
print(f"Erro RMS: {erro_rms:.2f}")
print(f"Erro percentual: {erro_percentual:.2f}%")
print(f"Correlação: {correlacao:.4f}\n")

# Atenuação
atenuacao_amplitude = np.max(np.abs(sinal_recebido)) / np.max(np.abs(sinal_emitido))
atenuacao_db = 20 * np.log10(atenuacao_amplitude)

# Ganho por frequência
pos_mask = freq > 0
freq_pos = freq[pos_mask]
H_mag = np.abs(H[pos_mask])
H_mag_db = 20 * np.log10(H_mag + 1e-10)

idx_central = np.argmin(np.abs(freq_pos - 2.85))
ganho_central_db = H_mag_db[idx_central]

# Graficos - Tempo
fig1, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(time, sinal_emitido, 'b-', linewidth=1, label='Emitido')
axes[0].plot(time, sinal_recebido, 'r-', linewidth=1, alpha=0.7, label='Recebido')
axes[0].set_title('Sinais Originais')
axes[0].set_ylabel('Amplitude')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, h, 'green', linewidth=1.2)
axes[1].set_title('Resposta Impulsiva h(t) - H(f)=Y(f)/X(f) com filtro passa-banda 1.5-4.5 MHz')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)

axes[2].plot(time, sinal_recebido, 'r-', linewidth=1.5, label='Real', alpha=0.8)
axes[2].plot(time, sinal_reconstruido, 'b--', linewidth=1.5, label='Reconstruído', alpha=0.8)
axes[2].set_title(f'Verificação (Correlação = {correlacao:.4f})')
axes[2].set_ylabel('Amplitude')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

axes[3].plot(time, erro, 'purple', linewidth=1)
axes[3].set_title(f'Erro (RMS = {erro_rms:.2f})')
axes[3].set_xlabel('Tempo (μs)')
axes[3].set_ylabel('Erro')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Graficos - Frequencia
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))

axes2[0, 0].plot(freq_pos, 20 * np.log10(np.abs(X[pos_mask]) + 1e-10), 'b-', linewidth=1.5)
axes2[0, 0].set_title('X(f): Sinal Emitido')
axes2[0, 0].set_ylabel('Magnitude (dB)')
axes2[0, 0].set_xlim([0, 15])
axes2[0, 0].axvline(2.85, color='red', linestyle='--', alpha=0.5)
axes2[0, 0].grid(True, alpha=0.3)

axes2[0, 1].plot(freq_pos, 20 * np.log10(np.abs(Y[pos_mask]) + 1e-10), 'r-', linewidth=1.5)
axes2[0, 1].set_title('Y(f): Sinal Recebido')
axes2[0, 1].set_ylabel('Magnitude (dB)')
axes2[0, 1].set_xlim([0, 15])
axes2[0, 1].axvline(2.85, color='red', linestyle='--', alpha=0.5)
axes2[0, 1].grid(True, alpha=0.3)

axes2[1, 0].plot(freq_pos, H_mag_db, 'green', linewidth=1.5)
axes2[1, 0].set_title('H(f) = Y(f)/X(f) com filtro passa-banda')
axes2[1, 0].set_xlabel('Frequência (MHz)')
axes2[1, 0].set_ylabel('Ganho (dB)')
axes2[1, 0].set_xlim([0, 15])
axes2[1, 0].axhline(0, color='gray', linestyle=':', alpha=0.5)
axes2[1, 0].axhline(atenuacao_db, color='orange', linestyle='--', alpha=0.7, linewidth=1.5,
                    label=f'Atenuação: {atenuacao_db:.1f} dB')
axes2[1, 0].axvline(1.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
axes2[1, 0].axvline(4.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
axes2[1, 0].legend()
axes2[1, 0].grid(True, alpha=0.3)

phase = np.angle(H[pos_mask])
axes2[1, 1].plot(freq_pos, phase, 'purple', linewidth=1.5)
axes2[1, 1].set_title('Fase de H(f)')
axes2[1, 1].set_xlabel('Frequência (MHz)')
axes2[1, 1].set_ylabel('Fase (rad)')
axes2[1, 1].set_xlim([0, 15])
axes2[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()