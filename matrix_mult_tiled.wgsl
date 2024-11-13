// ++++++++++++++++++++++++++++++
// ++++ Group 0 - parameters ++++
// ++++++++++++++++++++++++++++++
@group(0) @binding(0)
var<storage,read> m: array<f32>;

@group(0) @binding(1)
var<storage,read> n: array<f32>;

@group(0) @binding(2)
var<storage,read_write> p: array<f32>;

var<workgroup> mds: mat4x4<f32>;
var<workgroup> nds: mat4x4<f32>;

const TILE_WIDTH: i32 = 4;
const Width: i32 = 32;

@compute
@workgroup_size(wsx, wsy)
fn matrix_mult_tiled(@builtin(workgroup_id) blockIdx: vec3<u32>,
                     @builtin(local_invocation_id) threadIdx: vec3<u32>) {
    let bx: i32 = i32(blockIdx.x);
    let by: i32 = i32(blockIdx.y);
    let tx: i32 = i32(threadIdx.x);
    let ty: i32 = i32(threadIdx.y);

    let row: i32 = by * TILE_WIDTH + ty;
    let col: i32 = bx * TILE_WIDTH + tx;

    var Pvalue: f32 = 0.0;
    for(var ph: i32 = 0; ph < Width/TILE_WIDTH; ph++) {
        mds[ty][tx] = m[row*Width + ph*TILE_WIDTH + tx];
        nds[ty][tx] = n[(ph*TILE_WIDTH + ty)*Width + col];
        workgroupBarrier();

        for(var k: i32 = 0; k < TILE_WIDTH; k++) {
            Pvalue += mds[ty][k] * nds[k][tx];
        }
        workgroupBarrier();
    }

    p[row*Width + col] = Pvalue;
}