import wgpu.backends.wgpu_native
import numpy as np

# Cria as matrizes
width = 1024*8
tile_width = 32
m = np.random.rand(width, width).astype(np.float32)
n = np.random.rand(width, width).astype(np.float32)

# Multiplica no Python
p = np.linalg.matmul(m, n)

# Comeca a parte em WebGPU
# Pega uma referencia a GPU principal
device_gpu = wgpu.utils.get_default_device()
wsx = np.gcd(width, tile_width)
wsy = np.gcd(width, tile_width)

# Cria o shader para calculo contido no arquivo ``shader_2D_elast_cpml.wgsl''
with open('matrix_mult_tiled.wgsl') as shader_file:
    cshader_string = shader_file.read()
    cshader_string = cshader_string.replace('_wsx_', f'{wsx}')
    cshader_string = cshader_string.replace('_wsy_', f'{wsy}')
    cshader_string = cshader_string.replace('_width_', f'{width}')
    cshader_string = cshader_string.replace('_tilewidth_', f'{tile_width}')
    cshader_string = cshader_string.replace('_sizexds_', f'{tile_width*tile_width}')
    cshader = device_gpu.create_shader_module(code=cshader_string)

# Definicao dos buffers que terao informacoes compartilhadas entre CPU e GPU
buffer_m = device_gpu.create_buffer_with_data(data=m,
                                              usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_SRC)
buffer_n = device_gpu.create_buffer_with_data(data=n,
                                              usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_SRC)
buffer_p = device_gpu.create_buffer_with_data(data=np.zeros_like(p) - 2.0,
                                              usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
                                                    wgpu.BufferUsage.COPY_SRC)

# Cria o layout de binding
binding_layout = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.read_only_storage,
        },
    },
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.read_only_storage,
        },
    },
    {
        "binding": 2,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
]
bind_group_layout = device_gpu.create_bind_group_layout(entries=binding_layout)

# Liga os buffers com o layout (binding)
binding = [
    {
        "binding": 0,
        "resource": {"buffer": buffer_m, "offset": 0, "size": buffer_m.size},
    },
    {
        "binding": 1,
        "resource": {"buffer": buffer_n, "offset": 0, "size": buffer_n.size},
    },
    {
        "binding": 2,
        "resource": {"buffer": buffer_p, "offset": 0, "size": buffer_p.size},
    },
]
binding_group = device_gpu.create_bind_group(layout=bind_group_layout, entries=binding)

# Cria e ajusta o pipeline
pipeline_layout = device_gpu.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
compute_main = device_gpu.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": cshader,
             "entry_point": "matrix_mult_tiled",
             },
)

# Cria o decodificador de comandos
command_encoder = device_gpu.create_command_encoder()

# Coloca comandos para GPU na fila
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_bind_group(0, binding_group, [])
compute_pass.set_pipeline(compute_main)
dispatch_x = p.shape[0] // wsx
dispatch_y = p.shape[1] // wsy
compute_pass.dispatch_workgroups(dispatch_x, dispatch_y)
compute_pass.end()

# Envia comandos da fila para a GPU
device_gpu.queue.submit([command_encoder.finish()])

# Pega o resultado da GPU

result = np.array(device_gpu.queue.read_buffer(buffer_p).cast("f").tolist()).reshape(p.shape)
print(np.allclose(result, p))
