import numpy as np
import wgpu
import wgpu.backends.wgpu_native

data = np.array([1,1,1,1])
datatest= np.array([1])
t_out = np.zeros(shape=data.shape, dtype= np.int32)
device = wgpu.utils.get_default_device()

cshader = None
with open('test_max.wgsl') as shader_file:
    cshader_str = shader_file.read()
    cshader = device.create_shader_module(code=cshader_str)

#TAMANHO MAXIMO DO BUFFER

max_bf = device.create_buffer(size=268435456, usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
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

bif7 = device.create_buffer_with_data(data=datatest,
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

]

binding_layout7 = [
    {
        "binding": 7,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
]

binding_group = [
    {
        "binding": 0,
        "resource": {"buffer": max_bf, "offset": 0, "size": max_bf.size},
    },
    {
        "binding": 1,
        "resource": {"buffer": bif1, "offset": 0, "size": bif1.size},
    },
    {
        "binding": 2,
        "resource": {"buffer": bif2, "offset": 0, "size": bif2.size},
    },
    {
        "binding": 3,
        "resource": {"buffer": bif3, "offset": 0, "size": bif3.size},
    },
    {
        "binding": 4,
        "resource": {"buffer": bif4, "offset": 0, "size": bif4.size},
    },
    {
        "binding": 5,
        "resource": {"buffer": bif5, "offset": 0, "size": bif5.size},
    },
    {
        "binding": 6,
        "resource": {"buffer": bif6, "offset": 0, "size": bif6.size},
    },

]

binding_group7 = [
    {
        "binding": 7,
        "resource": {"buffer": bif7, "offset": 0, "size": bif7.size},
    },

]

bgl = device.create_bind_group_layout(entries=binding_layout)
bgl7 = device.create_bind_group_layout(entries=binding_layout7)
pipeline = device.create_pipeline_layout(bind_group_layouts=[bgl,bgl7])
bgf = device.create_bind_group(layout=bgl, entries=binding_group)
bgf7 = device.create_bind_group(layout=bgl7, entries=binding_group7)



# compute_big_t = device.create_compute_pipeline(
#     layout= pipeline,
#     compute= {"module":cshader, "entry_point":"big_t"},
# )

compute_incr_h = device.create_compute_pipeline(
    layout= pipeline,
    compute= {"module":cshader, "entry_point":"incr_h"},
)












