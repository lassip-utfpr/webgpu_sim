import numpy as np
import wgpu

wsx=2
wsy=3
data =  np.zeros((wsx, wsy), dtype=np.int32)
datatest= np.zeros((wsx, wsy), dtype=np.int32)
t_out = np.zeros(shape=data.shape, dtype= np.int32)

device = wgpu.utils.get_default_device()



with open('test_max.wgsl') as shader_file:
    cshader_str = shader_file.read().replace('wsx', f'{wsx}').replace('wsy', f'{wsy}')
    cshader = device.create_shader_module(code=cshader_str)

#DECLARAÇÃO DE BUFFERS
"""
MAIOR TAMANHO DE UM BUFFER
"""
max_bf0 = device.create_buffer(size=268435456, usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
                                                    wgpu.BufferUsage.COPY_SRC)

max_bf1 = device.create_buffer(size=268435456, usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
                                                    wgpu.BufferUsage.COPY_SRC)

max_bf2 = device.create_buffer(size=268435456, usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
                                                    wgpu.BufferUsage.COPY_SRC)

max_bf3 = device.create_buffer(size=268435456, usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
                                                    wgpu.BufferUsage.COPY_SRC)

max_bf4 = device.create_buffer(size=268435456, usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
                                                    wgpu.BufferUsage.COPY_SRC)

max_bf5 = device.create_buffer(size=268435456, usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
                                                    wgpu.BufferUsage.COPY_SRC)

max_bf6 = device.create_buffer(size=268435456, usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
                                                    wgpu.BufferUsage.COPY_SRC)

max_bf7 = device.create_buffer(size=268435456, usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
                                                    wgpu.BufferUsage.COPY_SRC)

max_bf8 = device.create_buffer(size=268435456, usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
                                                    wgpu.BufferUsage.COPY_SRC)

max_bf9 = device.create_buffer(size=268435456, usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
                                                    wgpu.BufferUsage.COPY_SRC)

"""
BUFFERS MENORES
"""

bf1x4 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf1 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf2 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf3= device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf4 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf5 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf6 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf7 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf8 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf9 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf10 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf11 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf12 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf13 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf14 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf15 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf16 = device.create_buffer_with_data(data=datatest, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)

bf17 = device.create_buffer_with_data(data=data, usage= wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)


#LAYOUTS DOS BINDINGS

b_layout_00 = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer":{
            "type": wgpu.BufferBindingType.storage,
        },
    },
]

b_layout_1 = [
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer":{
            "type": wgpu.BufferBindingType.storage,
        },
    },
]

b_layout_2 = [
    {
        "binding": 2,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer":{
            "type": wgpu.BufferBindingType.storage,
        },
    },
]

b_layout_3 = [
    {
        "binding": 3,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer":{
            "type": wgpu.BufferBindingType.storage,
        },
    },
]

b_layout_4 = [
    {
        "binding": 4,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer":{
            "type": wgpu.BufferBindingType.storage,
        },
    },
]

b_layout_5 = [
    {
        "binding": 5,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer":{
            "type": wgpu.BufferBindingType.storage,
        },
    },
]

b_layout_6 = [
    {"binding": nn,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
     } for nn in range(6,24)
]

b_layout_7 = [
    {"binding": ii,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
     } for ii in range(24,1001)
]

#LIGAÇÕES DOS BINDINGS

binding00 = [
    {
        "binding": 0,
        "resource": {"buffer": bf1x4, "offset": 0, "size": bf1x4.size},
    },
]

binding1 = [
    {
        "binding": 1,
        "resource": {"buffer": bf1x4, "offset": 0, "size": bf1x4.size},
    },
]

binding2 = [
    {
        "binding": 2,
        "resource": {"buffer": bf1x4, "offset": 0, "size": bf1x4.size},
    },
]

binding3 = [
    {
        "binding": 3,
        "resource": {"buffer": bf1x4, "offset": 0, "size": bf1x4.size},
    },
]

binding4 = [
    {
        "binding": 4,
        "resource": {"buffer": bf1x4, "offset": 0, "size": bf1x4.size},
    },
]

binding5 = [
    {
        "binding": 5,
        "resource": {"buffer": bf1x4, "offset": 0, "size": bf1x4.size},
    },
]

binding6 = [
    {
        "binding": 6,
        "resource": {"buffer": max_bf0, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 7,
        "resource": {"buffer": max_bf1, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 8,
        "resource": {"buffer": max_bf2, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 9,
        "resource": {"buffer": max_bf3, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 10,
        "resource": {"buffer": max_bf4, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 11,
        "resource": {"buffer": max_bf5, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 12,
        "resource": {"buffer": max_bf6, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 13,
        "resource": {"buffer": max_bf7, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 14,
        "resource": {"buffer": max_bf8, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 15,
        "resource": {"buffer": max_bf9, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 16,
        "resource": {"buffer": bf10, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 17,
        "resource": {"buffer": bf11, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 18,
        "resource": {"buffer": bf12, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 19,
        "resource": {"buffer": bf13, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 20,
        "resource": {"buffer": bf14, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 21,
        "resource": {"buffer": bf15, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 22,
        "resource": {"buffer": bf16, "offset": 0, "size": bf1x4.size},
    },
    {
        "binding": 23,
        "resource": {"buffer": bf17, "offset": 0, "size": bf1x4.size},
    },

]

binding7 = [
    {
        "binding": jj,
        "resource": {"buffer": bf7, "offset": 0, "size": bf1x4.size},
    } for jj in range(24, 1001)

]


#Juntar estrutura dos binding e criar o layout da pipeline

bind_l0 = device.create_bind_group_layout(entries=b_layout_00)
bind_l1 = device.create_bind_group_layout(entries=b_layout_1)
bind_l2 = device.create_bind_group_layout(entries=b_layout_2)
bind_l3 = device.create_bind_group_layout(entries=b_layout_3)
bind_l4 = device.create_bind_group_layout(entries=b_layout_4)
bind_l5 = device.create_bind_group_layout(entries=b_layout_5)
bind_l6 = device.create_bind_group_layout(entries=b_layout_6)
bind_l7 = device.create_bind_group_layout(entries=b_layout_7)

pipeline_l = device.create_pipeline_layout(bind_group_layouts=[bind_l0, bind_l1, bind_l2,bind_l3,bind_l4,bind_l5,bind_l6,bind_l7])

b_group_0 = device.create_bind_group(layout=bind_l0, entries=binding00)
b_group_1 = device.create_bind_group(layout=bind_l1, entries=binding1)
b_group_2 = device.create_bind_group(layout=bind_l2, entries=binding2)
b_group_3 = device.create_bind_group(layout=bind_l3, entries=binding3)
b_group_4 = device.create_bind_group(layout=bind_l4, entries=binding4)
b_group_5 = device.create_bind_group(layout=bind_l5, entries=binding5)
b_group_6 = device.create_bind_group(layout=bind_l6, entries=binding6)
b_group_7 = device.create_bind_group(layout=bind_l7, entries=binding7)

#Criando a pipeline e depois executando a mesma

compute_test = device.create_compute_pipeline(
    layout=pipeline_l,
    compute={"module": cshader, "entry_point": "testing"},
)

command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()

compute_pass.set_bind_group(0,b_group_0,[],0,999999)
compute_pass.set_bind_group(1,b_group_1,[],0,999999)
compute_pass.set_bind_group(2,b_group_2,[],0,999999)
compute_pass.set_bind_group(3,b_group_3,[],0,999999)
compute_pass.set_bind_group(4,b_group_4,[],0,999999)
compute_pass.set_bind_group(5,b_group_5,[],0,999999)
compute_pass.set_bind_group(6,b_group_6,[],0,999999)
compute_pass.set_bind_group(7,b_group_7,[],0,999999)

for i in range(wsy):
    compute_pass.set_pipeline(compute_test)
    compute_pass.dispatch_workgroups(wsx,wsy)
"""MAXIMO DESPACHAVEL"""

compute_pass.end()
device.queue.submit([command_encoder.finish()])

t_out = np.asarray(device.queue.read_buffer(bf17).cast("i")).reshape((wsx, wsy))
print(t_out)