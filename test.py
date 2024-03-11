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
]

binding_layout_1 = [
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },

]

binding_layout_2 = [
    {
        "binding": 2,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },

]

binding_layout_3 = [
    {
        "binding": 3,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },

]

binding_layout_4 = [
    {
        "binding": 4,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },

]

binding_layout_5 = [
    {
        "binding": 5,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },

]

binding_layout_6 = [
    {
        "binding": 6,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },

]



binding_layout_7 = [
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
#SIZE?

binding = [
    {
        "binding": 0,
        "resource": {"buffer": max_bf, "offset": 0, "size": 4},
    },
    ]

binding1 = [
    {
        "binding": 1,
        "resource": {"buffer": max_bf, "offset": 32, "size": 4},
    },
    ]
binding2 = [
    {
        "binding": 2,
        "resource": {"buffer": max_bf, "offset": 64, "size": 4},
    },
]
binding3 = [
    {
        "binding": 3,
        "resource": {"buffer": max_bf, "offset": 96, "size": 4},
    },
]

binding4 = [
    {
        "binding": 4,
        "resource": {"buffer": max_bf, "offset": 128, "size": 4},
    },
]

binding5 = [
    {
        "binding": 5,
        "resource": {"buffer": max_bf, "offset": 160, "size": 4},
    },
]

binding6 = [
    {
        "binding": 6,
        "resource": {"buffer": max_bf, "offset": 192, "size": 4},
    },
]

binding7 = [
{
        "binding": 7,
        "resource": {"buffer": bif7, "offset": 0, "size": 4},
    },
]

bgl = device.create_bind_group_layout(entries=binding_layout)
pipel = device.create_pipeline_layout(bind_group_layouts=[bgl])
bgf = device.create_bind_group(layout=bgl, entries=binding)
bgl1 = device.create_bind_group_layout(entries=binding_layout_1)
pipel1 = device.create_pipeline_layout(bind_group_layouts=[bgl1])
bgf1 = device.create_bind_group(layout=bgl1, entries= binding1)
bgl2 = device.create_bind_group_layout(entries=binding_layout_2)
pipel2 = device.create_pipeline_layout(bind_group_layouts=[bgl2])
bgf2 = device.create_bind_group(layout=bgl2, entries= binding2)
bgl3 = device.create_bind_group_layout(entries=binding_layout_3)
pipel3 = device.create_pipeline_layout(bind_group_layouts=[bgl3])
bgf3 = device.create_bind_group(layout=bgl3, entries= binding3)
bgl4 = device.create_bind_group_layout(entries=binding_layout_4)
pipel4 = device.create_pipeline_layout(bind_group_layouts=[bgl4])
bgf4= device.create_bind_group(layout=bgl4, entries= binding4)
bgl5 = device.create_bind_group_layout(entries=binding_layout_5)
pipel5 = device.create_pipeline_layout(bind_group_layouts=[bgl5])
bgf5 = device.create_bind_group(layout=bgl5, entries= binding5)
bgl6 = device.create_bind_group_layout(entries=binding_layout_6)
pipel6 = device.create_pipeline_layout(bind_group_layouts=[bgl6])
bgf6 = device.create_bind_group(layout=bgl6, entries= binding6)
bgl7 = device.create_bind_group_layout(entries=binding_layout_7)
pipel7 = device.create_pipeline_layout(bind_group_layouts=[bgl7])
bgf7 = device.create_bind_group(layout=bgl7, entries= binding7)

compute_big_t = device.create_compute_pipeline(
    layout = pipel,
    compute = {"module": cshader, "entry_point": "big_t"},
)

compute_incr_h = device.create_compute_pipeline(
    layout = pipel7,
    compute = {"module":cshader, "entry_point": "incr_h"}
)

command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()

compute_pass.set_bind_group(0, bgf, [], 0, 999999)
compute_pass.set_bind_group(1, bgf1, [], 0, 999999)
compute_pass.set_bind_group(2, bgf2, [], 0, 999999)
compute_pass.set_bind_group(3, bgf3, [], 0, 999999)
compute_pass.set_bind_group(4, bgf4, [], 0, 999999)
compute_pass.set_bind_group(5, bgf5, [], 0, 999999)
compute_pass.set_bind_group(6, bgf6, [], 0, 999999)
compute_pass.set_bind_group(7, bgf7, [], 0, 999999)


#compute_pass.set_pipeline(compute_big_t)
#compute_pass.dispatch_workgroups(65535) #MAXIMO SUPORTADO PARA DESPACHAR PARA O SHADER
for i in range(10):
    compute_pass.set_pipeline(compute_incr_h)
compute_pass.end()
device.queue.submit([command_encoder.finish()])

t_out =  device.queue.read_buffer(bif7).cast("i").tolist()

print(t_out)