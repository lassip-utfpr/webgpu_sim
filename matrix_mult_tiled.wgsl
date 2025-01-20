// +++++++++++++++++++++++++++++++
// +++ Index access functions ++++
// +++++++++++++++++++++++++++++++
// function to convert 2D [i,j] index into 1D [] index
fn ij(i: i32, j: i32, i_max: i32, j_max: i32) -> i32 {
    let index = j + i * j_max;

    return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max);
}

// ++++++++++++++++++++++++++++++
// ++++ Group 0 - parameters ++++
// ++++++++++++++++++++++++++++++
@group(0) @binding(0)
var<storage,read> m: array<f32>;

@group(0) @binding(1)
var<storage,read> n: array<f32>;

@group(0) @binding(2)
var<storage,read_write> p: array<f32>;

const TILE_WIDTH: i32 = _tilewidth_;
const Width: i32 = _width_;
var<workgroup> mds: array<f32, _sizexds_>;
var<workgroup> nds: array<f32, _sizexds_>;

@compute
@workgroup_size(_wsx_, _wsy_)
fn matrix_mult_tiled(@builtin(workgroup_id) blockIdx: vec3<u32>,
                     @builtin(local_invocation_id) threadIdx: vec3<u32>,
                     @builtin(local_invocation_index) index: u32) {
    let bx: i32 = i32(blockIdx.x);
    let by: i32 = i32(blockIdx.y);
    let tx: i32 = i32(threadIdx.x);
    let ty: i32 = i32(threadIdx.y);

    let row: i32 = by * TILE_WIDTH + ty;
    let col: i32 = bx * TILE_WIDTH + tx;

    var Pvalue: f32 = 0.0;
    for(var ph: i32 = 0; ph < Width/TILE_WIDTH; ph++) {
        mds[index] = m[row*Width + ph*TILE_WIDTH + tx];
        nds[index] = n[(ph*TILE_WIDTH + ty)*Width + col];
        workgroupBarrier();

        for(var k: i32 = 0; k < TILE_WIDTH; k++) {
            let mds_idx: i32 = ij(ty, k, TILE_WIDTH, TILE_WIDTH);
            let nds_idx: i32 = ij(k, tx, TILE_WIDTH, TILE_WIDTH);
            if(mds_idx >=0 && nds_idx >=0){
                Pvalue += mds[mds_idx] * nds[nds_idx];
            }
        }
        workgroupBarrier();
    }

    p[row*Width + col] = Pvalue;
}