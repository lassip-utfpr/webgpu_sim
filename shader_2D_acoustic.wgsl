struct LapIntValues {
    z_sz: i32,          // Z field size
    x_sz: i32,          // X field size
    z_src: i32,         // Z source
    x_src: i32,         // X source
    z_sens: i32,        // Z sensor
    x_sens: i32,        // X sensor
    num_coef: i32,      // num of discrete coefs
    k: i32              // iteraction
};

// Group 0 - parameters
@group(0) @binding(0)   // info_int buffer
var<storage,read_write> liv: LapIntValues;

@group(0) @binding(1) // info_float buffer
var<storage,read> coef: array<f32>;

@group(0) @binding(5) // source term
var<storage,read> src: array<f32>;

// Group 1 - simulation arrays
@group(1) @binding(2) // pressure field k
var<storage,read_write> PK: array<f32>;

@group(1) @binding(3) // pressure field k-1 
var<storage,read_write> PKm1: array<f32>;

@group(1) @binding(4) // pressure field k-2
var<storage,read_write> PKm2: array<f32>;

@group(1) @binding(8) // laplacian matrix
var<storage,read_write> lap: array<f32>;

@group(1) @binding(7) // velocity map
var<storage,read> c: array<f32>;

// Group 2 - sensors arrays
@group(2) @binding(6) // sensor signal
var<storage,read_write> sensor: array<f32>;

// function to convert 2D [z,x] index into 1D [zx] index
fn zx(z: i32, x: i32) -> i32 {
    let index = x + z * liv.x_sz;
    
    return select(-1, index, z >= 0 && z < liv.z_sz && x >= 0 && x < liv.x_sz);
}

// function to get an PK array value
fn getPK(z: i32, x: i32) -> f32 {
    let index: i32 = zx(z, x);
    
    return select(0.0, PK[index], index != -1);
}

// function to set a PK array value
fn setPK(z: i32, x: i32, val : f32) {
    let index: i32 = zx(z, x);
    
    if(index != -1) {
        PK[index] = val;
    }
}

// function to get an PKm1 array value
fn getPKm1(z: i32, x: i32) -> f32 {
    let index: i32 = zx(z, x);
    
    return select(0.0, PKm1[index], index != -1);
}

// function to set a PKm1 array value
fn setPKm1(z: i32, x: i32, val : f32) {
    let index: i32 = zx(z, x);
    
    if(index != -1) {
        PKm1[index] = val;
    }
} 

// function to get an PKm2 array value
fn getPKm2(z: i32, x: i32) -> f32 {
    let index: i32 = zx(z, x);
    
    return select(0.0, PKm2[index], index != -1);
}

// function to set a PKm2 array value
fn setPKm2(z: i32, x: i32, val : f32) {
    let index: i32 = zx(z, x);
    
    if(index != -1) {
        PKm2[index] = val;
    }
} 

// function to calculate laplacian
@compute
@workgroup_size(wsx, wsy)
fn laplacian(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);          // z thread index
    let x: i32 = i32(index.y);          // x thread index
    let num_coef: i32 = liv.num_coef;   // num coefs
    let idx: i32 = zx(z, x);
         
    // central
    if(idx != -1) {
        lap[idx] = 2.0 * coef[0] * getPKm1(z, x);

        for (var i = 1; i < num_coef; i = i + 1) {
            lap[idx] += coef[i] * (getPKm1(z - i, x) +  // i acima
                                   getPKm1(z + i, x) +  // i abaixo
                                   getPKm1(z, x - i) +  // i a esquerda
                                   getPKm1(z, x + i));  // i a direita
        }
    }
}

@compute
@workgroup_size(1)
fn incr_k() {
    liv.k += 1;
}

@compute
@workgroup_size(wsx, wsy)
fn sim(@builtin(global_invocation_id) index: vec3<u32>) {
    var add_src: f32 = 0.0;             // Source term
    let z: i32 = i32(index.x);          // z thread index
    let x: i32 = i32(index.y);          // x thread index
    let z_src: i32 = liv.z_src;         // source term z position
    let x_src: i32 = liv.x_src;         // source term x position
    let idx: i32 = zx(z, x);

    // --------------------
    // Update pressure field
    add_src = select(0.0, src[liv.k], z == z_src && x == x_src);
    setPK(z, x, -1.0*getPKm2(z, x) + 2.0*getPKm1(z, x) + c[idx]*c[idx]*lap[idx] + add_src);
        
    // --------------------
    // Circular buffer
    setPKm2(z, x, getPKm1(z, x));
    setPKm1(z, x, getPK(z, x));
    
    if(z == liv.z_sens && x == liv.x_sens) {
        sensor[liv.k] = getPK(z, x);
    }
}