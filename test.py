import numpy as np
import wgpu
import wgpu.backends.wgpu_native

data = np.array([1,1,1,1])
t_out = np.zeros(shape=data.shape, dtype= np.int32)
device = wgpu.utils.get_default_device()

cshader = None
with open('test_max.wgsl') as shader_file:
    cshader_str = shader_file.read()
    cshader = device.create_shader_module(code=cshader_str)

#TAMANHO MAXIMO DO BUFFER

max_bf = device.create_buffer(size=268435456, usage=wgpu.BufferUsage.STORAGE |
                                                        wgpu.BufferUsage.COPY_SRC)

bin0 = device.create_buffer_with_data(data=data,
                                            usage=wgpu.BufferUsage.STORAGE |
                                                  wgpu.BufferUsage.COPY_DST |
                                                  wgpu.BufferUsage.COPY_SRC)

bif1 = device.create_buffer(size = 256,
                                            usage=wgpu.BufferUsage.STORAGE |
                                                  wgpu.BufferUsage.COPY_DST |
                                                  wgpu.BufferUsage.COPY_SRC)

bif2 = device.create_buffer_with_data(data=data,
                                            usage=wgpu.BufferUsage.STORAGE |
                                                  wgpu.BufferUsage.COPY_DST |
                                                  wgpu.BufferUsage.COPY_SRC)

bif3= device.create_buffer_with_data(data=data,
                                            usage=wgpu.BufferUsage.STORAGE |
                                                  wgpu.BufferUsage.COPY_DST |
                                                  wgpu.BufferUsage.COPY_SRC)

bif4 = device.create_buffer_with_data(data=data,
                                            usage=wgpu.BufferUsage.STORAGE |
                                                  wgpu.BufferUsage.COPY_DST |
                                                  wgpu.BufferUsage.COPY_SRC)

bif5 = device.create_buffer_with_data(data=data,
                                            usage=wgpu.BufferUsage.STORAGE |
                                                  wgpu.BufferUsage.COPY_DST |
                                                  wgpu.BufferUsage.COPY_SRC)

bif6 = device.create_buffer_with_data(data=data,
                                            usage=wgpu.BufferUsage.STORAGE |
                                                  wgpu.BufferUsage.COPY_DST |
                                                  wgpu.BufferUsage.COPY_SRC)

bif7 = device.create_buffer_with_data(data=data,
                                            usage=wgpu.BufferUsage.STORAGE |
                                                  wgpu.BufferUsage.COPY_DST |
                                                  wgpu.BufferUsage.COPY_SRC)

binding_layout = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
    {
        "binding": 2,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
    {
        "binding": 3,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
{
        "binding": 4,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
    {
        "binding": 5,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
    {
        "binding": 6,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
    {
        "binding": 7,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
]

#O BINDING 0 ESTÁ PEGANDO O MAX_BF INTEIRO???
#COMO FAZ PARA LIMITAR QUANTO "OFFSET" UM BINDING PEGA?

binding = [
    {
        "binding": 0,
        "resource": {"buffer": max_bf, "offset": 0, "size": 1},
    },
    {
        "binding": 1,
        "resource": {"buffer": bif1, "offset": 0, "size": 1},
    },
    {
        "binding": 2,
        "resource": {"buffer": bif1, "offset": 32, "size": 1},
    },
    {
        "binding": 3,
        "resource": {"buffer": max_bf, "offset": 32, "size": 1},
    },
    {
        "binding": 4,
        "resource": {"buffer": max_bf, "offset": 64, "size": 1},
    },
    {
        "binding": 5,
        "resource": {"buffer": max_bf, "offset": 96, "size": 1},
    },
    {
        "binding": 6,
        "resource": {"buffer": max_bf, "offset": 128, "size": 1},
    },
    {
        "binding": 7,
        "resource": {"buffer": max_bf, "offset": 160, "size": 1},
    },

]

bgl = device.create_bind_group_layout(entries=binding_layout)
pipel = device.create_pipeline_layout(bind_group_layouts=[bgl])
bgf = device.create_bind_group(layout=bgl, entries=binding)

compute_big_t = device.create_compute_pipeline(
    layout = pipel,
    compute = {"module": cshader, "entry_point": "big_t"},
)