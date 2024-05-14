import wgpu
import numpy as np
from numpy import pi
import ast
from scipy.signal import gausspulse
from simul_utils import SimulationROI
import matplotlib.pyplot as plt

# Configurações Iniciais
flt32 = np.float32
vx_min = -100000.0
vx_max = 100000.0
device_gpu = wgpu.utils.get_default_device()
HUGEVAL = 1.0e30  # Mesmo valor adotado pelo professor
STABILITY_THRESHOLD = 1.0e25  # Limite para considerar que a simulacao esta instavel




with open('config.json', 'r') as f:
    configs = ast.literal_eval(f.read())
    data_src = np.array(configs["sources"])
    data_rec = np.array(configs["receivers"])
    source_term_cfg = configs["source_term"]

    simul_roi = SimulationROI(**configs["roi"], pad=1)
    wavenumber_x = (2.0 * pi * source_term_cfg["freq"]) / configs["specimen_params"]["cp"]


# Parametros da simulacao
nx = simul_roi.get_nx()

# Escala do grid (valor do passo no espaco em milimetros)
dx = simul_roi.w_step
one_dx = 1.0 / dx

# Velocidades do som e densidade do meio
cp = configs["specimen_params"]["cp"]  # [mm/us]
cs = configs["specimen_params"]["cs"]  # [mm/us]
rho = configs["specimen_params"]["rho"]
mu = rho * cs * cs
lambda_ = rho * (cp * cp - 2.0 * cs * cs)
lambdaplus2mu = rho * cp * cp

# Numero total de passos de tempo
NSTEP = configs["simul_params"]["time_steps"]

# Passo de tempo em microssegundos
dt = configs["simul_params"]["dt"]

# Define a posicao das fontes
NSRC = data_src.shape[0]
i_src = np.array([simul_roi.get_nearest_grid_idx(p[0:3]) for p in data_src])
ix_src = i_src[:, 0].astype(np.int32)
aux_src = ix_src[0]

# Parametros da fonte
f0 = source_term_cfg["freq"]  # frequencia [MHz]
t0 = data_src[:, 3].reshape((1, NSRC))
factor = flt32(source_term_cfg["gain"])
t = np.expand_dims(np.arange(NSTEP) * dt, axis=1)

# Gauss pulse
source_term = factor * flt32(gausspulse((t - t0), fc=f0, bw=source_term_cfg["bw"])).astype(flt32)
# Define a localizacao dos receptores
NREC = data_rec.shape[0]
i_rec = np.array([simul_roi.get_nearest_grid_idx(p[0:3]) for p in data_rec])
ix_rec = i_rec[:, 0].astype(np.int32)

# Valor da potencia para calcular "d0"
NPOWER = 2.0

vx = np.zeros(nx, dtype=flt32)
sigmaxx = np.zeros(nx, dtype=flt32)
sisvx = np.zeros((NSTEP, NREC), dtype=flt32)

wsx = nx

def sim_1D_wgpu(device):
    global source_term
    global vx, sigmaxx
    global ix_rec
    global sisvx

    # Arrays com parametros
    params_i32 = np.array([nx, aux_src, NSTEP, NREC, 0], dtype=np.int32)
    params_f32 = np.array([dx, dt, rho, lambda_, mu, lambdaplus2mu], dtype=flt32)

    # Rodar shader
    with open('shader_1D_elast.wgsl') as shader_file:
        cshader_string = shader_file.read()
        cshader_string = cshader_string.replace('wsx', f'{wsx}')
        cshader = device.create_shader_module(code=cshader_string)

    # Buffers

    # Binding 0
    bf_param_flt32 = device.create_buffer_with_data(data=params_f32, usage=wgpu.BufferUsage.STORAGE |
                                                                          wgpu.BufferUsage.COPY_SRC)
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU

    # Forcas da fonte
    # Binding 1
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    bf_src = device.create_buffer_with_data(data=np.column_stack((source_term)),
                                             usage=wgpu.BufferUsage.STORAGE |
                                                   wgpu.BufferUsage.COPY_SRC)

    # Binding 2
    bf_param_int32 = device.create_buffer_with_data(data=params_i32, usage=wgpu.BufferUsage.STORAGE |
                                                                          wgpu.BufferUsage.COPY_SRC)

    # Binding 3
    bf_vx = device.create_buffer_with_data(data=np.vstack(vx), usage=wgpu.BufferUsage.STORAGE |
                                                                                wgpu.BufferUsage.COPY_DST |
                                                                                wgpu.BufferUsage.COPY_SRC)

    # Binding 4
    bf_sigma = device.create_buffer_with_data(data=np.vstack(sigmaxx),
                                           usage=wgpu.BufferUsage.STORAGE |
                                                 wgpu.BufferUsage.COPY_DST |
                                                 wgpu.BufferUsage.COPY_SRC)

    # Binding 5
    bf_sensx = device.create_buffer_with_data(data=sisvx, usage=wgpu.BufferUsage.STORAGE |
                                                                wgpu.BufferUsage.COPY_DST |
                                                                wgpu.BufferUsage.COPY_SRC)

    # Binding 6
    bf_pos_x = device.create_buffer_with_data(data=ix_rec, usage=wgpu.BufferUsage.STORAGE |
                                                                     wgpu.BufferUsage.COPY_SRC)


    # Binding Layouts

    bl_params = [
        {"binding": 0,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         },
        {"binding": 1,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         },
        {
            "binding": 2,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,}
        }
    ]

    bl_sim = [
        {"binding": 3,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
        {"binding": 4,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         }
    ]

    bl_sensors = [
        {"binding": 5,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
        {"binding": 6,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         }
    ]


    # Bindings

    b_params = [
        {
            "binding": 0,
            "resource": {"buffer": bf_param_flt32, "offset": 0, "size": bf_param_flt32.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": bf_src, "offset": 0, "size": bf_src.size},
        },
        {
            "binding": 2,
            "resource": {"buffer": bf_param_int32, "offset": 0, "size": bf_param_int32.size},
        },
    ]
    b_sim = [
        {
            "binding": 3,
            "resource": {"buffer": bf_vx, "offset": 0, "size": bf_vx.size},
        },
        {
            "binding": 4,
            "resource": {"buffer": bf_sigma, "offset": 0, "size": bf_sigma.size},
        },
    ]
    b_sensors = [
        {
            "binding": 5,
            "resource": {"buffer": bf_sensx, "offset": 0, "size": bf_sensx.size},
        },
        {
            "binding": 6,
            "resource": {"buffer": bf_pos_x, "offset": 0, "size": bf_pos_x.size},
        },
    ]

    # B Layouts + Bindings + Pipeline Layout
    bgl_0 = device.create_bind_group_layout(entries=bl_params)
    bgl_1 = device.create_bind_group_layout(entries=bl_sim)
    bgl_2 = device.create_bind_group_layout(entries=bl_sensors)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bgl_0, bgl_1, bgl_2])
    bg_0 = device.create_bind_group(layout=bgl_0, entries=b_params)
    bg_1 = device.create_bind_group(layout=bgl_1, entries=b_sim)
    bg_2 = device.create_bind_group(layout=bgl_2, entries=b_sensors)

    # Pipeline config
    compute_sigmax = device.create_compute_pipeline(layout=pipeline_layout,
                                                          compute={"module": cshader, "entry_point": "sigmax"})
    compute_velx = device.create_compute_pipeline(layout=pipeline_layout,
                                                             compute={"module": cshader,
                                                                      "entry_point": "velx"})
    compute_sensx = device.create_compute_pipeline(layout=pipeline_layout,
                                                              compute={"module": cshader,
                                                                       "entry_point": "sensx"})
    compute_incr_it = device.create_compute_pipeline(layout=pipeline_layout,
                                                            compute={"module": cshader,
                                                                     "entry_point": "incr_it"})

    for it in range(1, NSTEP + 1):

        command_encoder = device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()

        compute_pass.set_bind_group(0, bg_0, [], 0, 999999)
        compute_pass.set_bind_group(1, bg_1, [], 0, 999999)
        compute_pass.set_bind_group(2, bg_2, [], 0, 999999)

        compute_pass.set_pipeline(compute_sigmax)
        compute_pass.dispatch_workgroups(wsx)

        compute_pass.set_pipeline(compute_velx)
        compute_pass.dispatch_workgroups(wsx)

        compute_pass.set_pipeline(compute_sensx)
        compute_pass.dispatch_workgroups(wsx)

        compute_pass.set_pipeline(compute_incr_it)
        compute_pass.dispatch_workgroups(1)

        compute_pass.end()
        device.queue.submit([command_encoder.finish()])



    # Resultados
    vxgpu = np.asarray(device.queue.read_buffer(bf_vx,buffer_offset=0, size=vx.size * 4).cast("f")).reshape(nx)
    sens_vx = np.array(device.queue.read_buffer(bf_sensx).cast("f")).reshape((NSTEP, NREC))
    return vxgpu, sens_vx

vx, sisvx = sim_1D_wgpu(device_gpu)

# Inicializa a figura que apresenta a simulacao em tempo real
ix_min = simul_roi.get_ix_min()
ix_max = simul_roi.get_ix_max()
fig, ax = plt.subplots()
ax.plot(simul_roi.w_points, vx[ix_min:ix_max])
ax.set_xlim(0.0, dx * nx)
ax.set_ylim(vx_min, vx_max)

# Plota o sinal da fonte
plt.figure()
plt.plot(t, source_term)
plt.title('Sinal da fonte')

# Plota as velocidades tomadas no sensores
for irec in range(NREC):
    fig, ax = plt.subplots(1, sharex=True, sharey=True)
    fig.suptitle(f'Receptor {irec + 1} [GPU]')
    ax.plot(sisvx[:, irec])
    ax.set_title(r'$V_x$')

plt.show()