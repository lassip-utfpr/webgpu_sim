import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl
from pylab import cm
from matplotlib import rc
import warnings
warnings.filterwarnings("ignore") # igora warning the arquivo vazio

# plots style configuration
rc('text', usetex=True)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
colors = cm.get_cmap('tab10', 3)

show_results = False
plot_results = True
save_combined_results = False
save_plots = True
# resultados para incremento no numero de pontos temporais com grid fixa
temporal_points_test = False


if temporal_points_test:
    # Extrai os dados de tempo da GPU da NVIDIA (Andre - Data: 2022-11-26)
    files_nvidia_cpu = glob.glob("./results/result_20221126*CPU.csv")
    files_nvidia_gpu = glob.glob("./results/result_20221126*GPU.csv")
    # files_nvidia_desc =glob.glob("./results/result_20221126*desc.txt")

else:
    # Extrai os dados de tempo da GPU da NVIDIA (Giovanni - Data: 2022-11-19)
    files_nvidia_cpu = glob.glob("./results/result_20221119*CPU.csv")
    files_nvidia_gpu = glob.glob("./results/result_20221119*GPU.csv")
    # files_nvidia_desc =glob.glob("./results/result_20221119*desc.txt")

    # Extrai os dados de tempo da GPU da INTEL (Giovanni - Data: 2022-11-22)
    files_intel_cpu = glob.glob("./results/result_20221122*CPU.csv")
    files_intel_gpu = glob.glob("./results/result_20221122*GPU.csv")
    # files_intel_desc = glob.glob("./results/result_20221122*desc.txt")



# PARA DADOS NVIDIA
nvidia_tsim_gpu = np.zeros((len(files_nvidia_gpu), 2))
nvidia_tsim_cpu = np.zeros((len(files_nvidia_cpu), 2))

for idx, file in enumerate(files_nvidia_gpu):
    if temporal_points_test:
        # resultados para incremento do num pontos no tempo com grid fixa
        N = str(file).split("_")[3]  # num pontos no tempo
    else:
        # resultados para incremento do num pixels com pontos no tempo fixo
        N = str(file).split("_")[2].split('x')[0]  # num pixels
    sim_data = np.genfromtxt(f"{file}", delimiter=',') # tempo de cada sim
    if sim_data.size:
        # tempo médio total
        if len(sim_data) > 5:
            t_sim = sim_data[5:].mean()
        else:
            t_sim = sim_data.mean()
    else:
        t_sim = -1
    # armazena os dados
    nvidia_tsim_gpu[idx, 0] = N
    nvidia_tsim_gpu[idx, 1] = t_sim

for idx, file in enumerate(files_nvidia_cpu):
    if temporal_points_test:
        # resultados para incremento do num pontos no tempo com grid fixa
        N = str(file).split("_")[3]  # num pontos no tempo
    else:
        # resultados para incremento do num pixels com pontos no tempo fixo
        N = str(file).split("_")[2].split('x')[0]  # num pixels
    sim_data = np.genfromtxt(f"{file}", delimiter=',') # tempo de cada sim
    if sim_data.size:
        # tempo médio total
        if len(sim_data) > 5:
            t_sim = sim_data[5:].mean()
        else:
            t_sim = sim_data.mean()
    else:
        t_sim = -1
    # armazena os dados
    nvidia_tsim_cpu[idx, 0] = N
    nvidia_tsim_cpu[idx, 1] = t_sim

# ordena os dados por ordem crescente de numero de pixels
nvidia_tsim_cpu = nvidia_tsim_cpu[np.argsort(nvidia_tsim_cpu[:, 0])]
nvidia_tsim_gpu = nvidia_tsim_gpu[np.argsort(nvidia_tsim_gpu[:, 0])]
# concatena dados cpu + gpu
# col 1 - N / col 2 - CPU / col 3 - GPU
nvidia_tsim = np.delete(np.concatenate((nvidia_tsim_cpu, nvidia_tsim_gpu), axis=1), 2, 1)
# Elimina dados negativos (-1) que representam a não realização de simulação
nvidia_tsim_cpu = nvidia_tsim_cpu[nvidia_tsim_cpu.min(axis=1)>=0, :]
nvidia_tsim_gpu = nvidia_tsim_gpu[nvidia_tsim_gpu.min(axis=1)>=0, :]
nvidia_tsim = nvidia_tsim[nvidia_tsim.min(axis=1)>=0, :]

# =============================================
# PARA DADOS INTEL
# não tem teste com incremento de pontos temporais

if not temporal_points_test:
    intel_tsim_gpu = np.zeros((len(files_intel_gpu), 2))
    intel_tsim_cpu = np.zeros((len(files_intel_cpu), 2))

    for idx, file in enumerate(files_intel_gpu):
        if temporal_points_test:
            # resultados para incremento do num pontos no tempo com grid fixa
            N = str(file).split("_")[3]  # num pontos no tempo
        else:
            # resultados para incremento do num pixels com pontos no tempo fixo
            N = str(file).split("_")[2].split('x')[0]  # num pixels
        sim_data = np.genfromtxt(f"{file}", delimiter=',') # tempo de cada sim
        if sim_data.size:
            # tempo médio total
            if len(sim_data) > 5:
                t_sim = sim_data[5:].mean()
            else:
                t_sim = sim_data.mean()
        else:
            t_sim = -1
        # armazena os dados
        intel_tsim_gpu[idx, 0] = N
        intel_tsim_gpu[idx, 1] = t_sim

    for idx, file in enumerate(files_intel_cpu):
        if temporal_points_test:
            # resultados para incremento do num pontos no tempo com grid fixa
            N = str(file).split("_")[3]  # num pontos no tempo
        else:
            # resultados para incremento do num pixels com pontos no tempo fixo
            N = str(file).split("_")[2].split('x')[0]  # num pixels
        sim_data = np.genfromtxt(f"{file}", delimiter=',') # tempo de cada sim
        if sim_data.size:
            # tempo médio total
            if len(sim_data) > 5:
                t_sim = sim_data[5:].mean()
            else:
                t_sim = sim_data.mean()
        else:
            t_sim = -1
        # armazena os dados
        intel_tsim_cpu[idx, 0] = N
        intel_tsim_cpu[idx, 1] = t_sim

    # ordena os dados por ordem crescente de numero de pixels
    intel_tsim_cpu = intel_tsim_cpu[np.argsort(intel_tsim_cpu[:, 0])]
    intel_tsim_gpu = intel_tsim_gpu[np.argsort(intel_tsim_gpu[:, 0])]
    # concatena dados cpu + gpu
    # col 1 - N / col 2 - CPU / col 3 - GPU
    intel_tsim = np.delete(np.concatenate((intel_tsim_cpu, intel_tsim_gpu), axis=1), 2, 1)
    # Elimina dados negativos (-1) que representam a não realização de simulação
    intel_tsim_cpu = intel_tsim_cpu[intel_tsim_cpu.min(axis=1)>=0, :]
    intel_tsim_gpu = intel_tsim_gpu[intel_tsim_gpu.min(axis=1)>=0, :]
    intel_tsim = intel_tsim[intel_tsim.min(axis=1)>=0, :]

if plot_results:
    if temporal_points_test:
        cpu_sim_result = plt.figure()
        plt.title(f'CPU simulation for 256 x 256 grid')
        plt.plot(nvidia_tsim[:, 0], nvidia_tsim_cpu[:, 1])
        plt.xlabel('Number of time points')
        plt.ylabel('Time (s)')
        plt.grid()

        nvidia_sim_result = plt.figure()
        plt.title(f'NVIDIA GPU simulation for 256 x 256 grid')
        plt.plot(nvidia_tsim_gpu[:, 0], nvidia_tsim_gpu[:, 1])
        plt.xlabel('Number of time points')
        plt.ylabel('Time (s)')
        plt.grid()

        cpu_vs_nvidia_sim_result = plt.figure()
        plt.title(f'CPU vs NVIDIA GPU simulation for 256 x 256 grid')
        plt.plot(nvidia_tsim[:, 0], nvidia_tsim[:, 1], label="CPU", color=colors(0))
        plt.plot(nvidia_tsim[:, 0], nvidia_tsim[:, 2], label="NVIDIA GPU", color=colors(1))
        plt.legend(frameon=False, loc='lower right')
        plt.yscale("log")
        plt.xlabel('Number of time points')
        plt.ylabel('Time (s)')
        plt.grid()

        cpu_vs_nvidia_speedup_result = plt.figure()
        plt.title(f'CPU vs NVIDIA GPU speed up for 256 x 256 grid')
        plt.plot(nvidia_tsim[:, 0], nvidia_tsim[:, 1] / nvidia_tsim[:, 2])
        plt.xlabel('Number of time points')
        plt.ylabel('Speed up')
        plt.grid()
    else:
        cpu_vs_gpu_roi_result = plt.figure()
        plt.plot(nvidia_tsim[:, 0]**2 , nvidia_tsim[:, 1], label="CPU", color=colors(1), marker="s")
        plt.plot(nvidia_tsim[:, 0] ** 2, nvidia_tsim[:, 2], label="NVIDIA", color=colors(0), marker="o")
        plt.plot(intel_tsim_gpu[:, 0][:17] ** 2, intel_tsim_gpu[:, 1][:17], label="Intel", color=colors(2), marker="^")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(which='minor', axis='both', c='0.95')
        plt.legend(frameon=False, loc='lower right')
        plt.xlabel('Number of pixels')
        plt.ylabel('Time (s)')
        plt.grid()

        cpu_vs_gpu_speedup_result = plt.figure()
        plt.plot(nvidia_tsim[:, 0] ** 2, nvidia_tsim[:, 1]/nvidia_tsim[:, 2], label="NVIDIA", color=colors(1),
                 marker="o")
        plt.plot(intel_tsim_gpu[:, 0][:17] ** 2, nvidia_tsim[:, 1] / intel_tsim_gpu[:, 1][:17], label="Intel",
                 color=colors(0), marker="^")
        plt.xscale('log')
        plt.legend(frameon=False, loc='upper left')
        plt.xlabel('Number of pixels')
        plt.ylabel('Speedup')
        plt.grid(which='minor', axis='x', c='0.95')
        plt.grid()

        # cpu_sim_result = plt.figure()
        # plt.title(f'1.000 temporal points CPU simulation')
        # plt.plot(nvidia_tsim[:, 0]**2, nvidia_tsim_cpu[:, 1])
        # plt.xlabel('Number of pixels')
        # plt.ylabel('Time (s)')
        # plt.grid()
        #
        # nvidia_sim_result = plt.figure()
        # plt.title(f'1.000 temporal points NVIDIA GPU simulation')
        # plt.plot(nvidia_tsim_gpu[:, 0]**2, nvidia_tsim_gpu[:, 1])
        # plt.xlabel('Number of pixels')
        # plt.ylabel('Time (s)')
        # plt.grid()
        #
        # cpu_vs_nvidia_sim_result = plt.figure()
        # plt.title(f'1.000 temporal points CPU vs NVIDIA GPU simulation')
        # plt.plot(nvidia_tsim[:, 0]**2, nvidia_tsim[:, 1], label="CPU", color=colors(0))
        # plt.plot(nvidia_tsim[:, 0]**2, nvidia_tsim[:, 2], label="NVIDIA GPU", color=colors(1))
        # plt.legend(frameon=False, loc='lower right')
        # plt.yscale("log")
        # plt.xlabel('Number of pixels')
        # plt.ylabel('Time (s)')
        # plt.grid()
        #
        # cpu_vs_nvidia_speedup_result = plt.figure()
        # plt.title(f'1.000 temporal points CPU vs NVIDIA GPU speed up')
        # plt.plot(nvidia_tsim[:, 0]**2, nvidia_tsim[:, 1]/nvidia_tsim[:, 2])
        # plt.xlabel('Number of pixels')
        # plt.ylabel('Speed up')
        # plt.grid()
        #
        # intel_sim_result = plt.figure()
        # plt.title(f'1.000 temporal points Intel GPU simulation')
        # plt.plot(intel_tsim_gpu[:, 0]**2, intel_tsim_gpu[:, 1])
        # plt.xlabel('Number of pixels')
        # plt.ylabel('Time (s)')
        # plt.grid()
        #
        # cpu_vs_intel_sim_result = plt.figure()
        # plt.title(f'1.000 temporal points CPU vs Intel GPU simulation')
        # plt.plot(nvidia_tsim[:, 0]**2, nvidia_tsim[:, 1], label="CPU", color=colors(0))  # CPU não importa a "GPU"
        # plt.plot(intel_tsim_gpu[:, 0][:17]**2, intel_tsim_gpu[:, 1][:17], label="Intel GPU", color=colors(1))
        # plt.legend(frameon=False, loc='lower right')
        # plt.yscale("log")
        # plt.xlabel('Number of pixels')
        # plt.ylabel('Time (s)')
        # plt.grid()
        #
        # cpu_vs_intel_speedup_result = plt.figure()
        # plt.title(f'1.000 temporal points CPU vs Intel GPU speed up')
        # plt.plot(nvidia_tsim[:, 0]**2, nvidia_tsim[:, 1] / intel_tsim_gpu[:, 1][:17])
        # plt.xlabel('Number of pixels')
        # plt.ylabel('Speed up')
        # plt.grid()

    if show_results:
        plt.show()


if save_combined_results:
    now = datetime.now()
    if temporal_points_test:
        name = f'combined_results/cpu_vs_nvidia_tpoints_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt'
        np.savetxt(name, nvidia_tsim, '%10.3f', delimiter=',')

        name = f'combined_results/nvidia_tpoints_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt'
        np.savetxt(name, nvidia_tsim_gpu, '%10.3f', delimiter=',')

        name = f'combined_results/cpu_tpoints_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt'
        np.savetxt(name, nvidia_tsim_cpu, '%10.3f', delimiter=',')
    else:
        name = f'combined_results/cpu_vs_nvidia_pixels_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt'
        np.savetxt(name, nvidia_tsim, '%10.3f', delimiter=',')

        name = f'combined_results/nvidia_pixels_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt'
        np.savetxt(name, nvidia_tsim_gpu, '%10.3f', delimiter=',')

        name = f'combined_results/cpu_pixels_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt'
        np.savetxt(name, nvidia_tsim_cpu, '%10.3f', delimiter=',')

        name = f'combined_results/cpu_vs_intel_pixels_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt'
        np.savetxt(name, intel_tsim, '%10.3f', delimiter=',')

        name = f'combined_results/intel_pixels_{now.strftime("%Y-%m-%d_%H-%M-%S")}.txt'
        np.savetxt(name, intel_tsim_gpu, '%10.3f', delimiter=',')

if save_plots:
    if plot_results:
        now = datetime.now()
        if temporal_points_test:
            cpu_sim_result.savefig(f'time_plots/cpu_tpoints_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
            nvidia_sim_result.savefig(f'time_plots/nvidia_tpoints_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
            cpu_vs_nvidia_sim_result.savefig(f'time_plots/cpu_vs_nvidia_tpoints_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
            cpu_vs_nvidia_speedup_result.savefig(f'time_plots/cpu_vs_nvidia_tpoints_speedup_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
        else:
            cpu_vs_gpu_roi_result.savefig(f'time_plots/cpu_vs_gpu_roi_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
            cpu_vs_gpu_speedup_result.savefig(f'time_plots/cpu_vs_gpu_speedup_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
            # cpu_sim_result.savefig(f'time_plots/cpu_pixels_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
            # nvidia_sim_result.savefig(f'time_plots/nvidia_pixels_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
            # cpu_vs_nvidia_sim_result.savefig(f'time_plots/cpu_vs_nvidia_pixels_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
            # cpu_vs_nvidia_speedup_result.savefig(f'time_plots/cpu_vs_nvidia_pixels_speedup_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
            # intel_sim_result.savefig(f'time_plots/intel_pixels_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
            # cpu_vs_intel_sim_result.savefig(f'time_plots/cpu_vs_intel_pixels_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
            # cpu_vs_intel_speedup_result.savefig(f'time_plots/cpu_vs_intel_pixels_speedup_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')




