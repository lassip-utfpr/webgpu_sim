import wgpu
if wgpu.version_info[1] > 11:
    import wgpu.backends.wgpu_native  # Select backend 0.13.X
else:
    import wgpu.backends.rs  # Select backend 0.9.5

# Create device and shader object
device = wgpu.utils.get_default_device()

# Create input data as a memoryview
n = 4
data = memoryview(bytearray(n * 4)).cast("i")
for i in range(n):
    data[i] = i

# Get a GPU buffer in a mapped state and an arrayBuffer for writing
gpu_write_buffer = device.create_buffer_with_data(data=data,
                                                  usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC)

# Get a GPU buffer for reading in an unmapped state
gpu_read_buffer = device.create_buffer(size=data.nbytes,
                                       usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)

# Encode commands for copying buffer to buffer
copy_encoder = device.create_command_encoder()
copy_encoder.copy_buffer_to_buffer(gpu_write_buffer, 0, gpu_read_buffer, 0, data.nbytes)

# Submit copy commands.
copy_commands = copy_encoder.finish()
device.queue.submit([copy_commands])

# Read buffer
if wgpu.version_info[1] > 11:
    gpu_read_buffer.map(wgpu.MapMode.READ)
    read_data = gpu_read_buffer.read_mapped().cast("I")
else:
    read_data = gpu_read_buffer.map_read().cast("I")
assert read_data.tolist() == data.tolist()

print(read_data.tolist())
