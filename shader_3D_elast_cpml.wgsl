struct SimIntValues {
    x_sz: i32,          // x field size
    y_sz: i32,          // y field size
    z_sz: i32,          // z field size
    n_iter: i32,        // num iterations
    n_probe_elem: i32,  // num probe elements
    n_src: i32,         // num sources points
    n_rec: i32,         // num receptors points
    fd_coeff: i32,      // num fd coefficients
    it: i32             // time iteraction
};

struct SimFltValues {
    cp: f32,            // longitudinal sound speed
    cs: f32,            // transverse sound speed
    dx: f32,            // delta x
    dy: f32,            // delta y
    dz: f32,            // delta z
    dt: f32,            // delta t
    rho: f32,           // density
    lambda: f32,        // Lame parameter
    mu : f32,           // Lame parameter
    lambdaplus2mu: f32  // Lame parameter
};

// Group 0 - parameters
@group(0) @binding(0)   // param_flt32
var<storage,read> sim_flt_par: SimFltValues;

@group(0) @binding(1) // source term
var<storage,read> source_term: array<f32>;

@group(0) @binding(27) // source term index
var<storage,read> idx_src: array<i32>;

@group(0) @binding(2) // a_x, b_x, k_x, a_x_h, b_x_h, k_x_h
var<storage,read> coef_x: array<f32>;

@group(0) @binding(3) // a_y, b_y, k_y, a_y_h, b_y_h, k_y_h
var<storage,read> coef_y: array<f32>;

@group(0) @binding(4) // a_z, b_z, k_z, a_z_h, b_z_h, k_z_h
var<storage,read> coef_z: array<f32>;

@group(0) @binding(5) // param_int32
var<storage,read_write> sim_int_par: SimIntValues;

@group(0) @binding(25) // idx_fd
var<storage,read> idx_fd: array<i32>;

@group(0) @binding(28) // fd_coeff
var<storage,read> fd_coeffs: array<f32>;

// Group 1 - simulation arrays
@group(1) @binding(6) // velocity fields (vx, vy, vz)
var<storage,read_write> vel: array<f32>;

@group(1) @binding(7) // normal stress fields (sigmaxx, sigmayy, sigmazz)
var<storage,read_write> sig_norm: array<f32>;

@group(1) @binding(8) // transversal stress fields (sigmaxy, sigmaxz, sigmayz)
var<storage,read_write> sig_trans: array<f32>;

@group(1) @binding(9) // memory fields
                      // memory_dvx_dx, memory_dvx_dy, memory_dvx_dz
var<storage,read_write> memo_v_dx: array<f32>;

@group(1) @binding(10) // memory fields
                      // memory_dvx_dy, memory_dvy_dy, memory_dvz_dy
var<storage,read_write> memo_v_dy: array<f32>;

@group(1) @binding(11) // memory fields
                      // memory_dvx_dz, memory_dvy_dz, memory_dvz_dz
var<storage,read_write> memo_v_dz: array<f32>;

@group(1) @binding(12) // memory fields
                      // memory_dsigmaxx_dx, memory_dsigmaxy_dy, memory_dsigmaxz_dz
var<storage,read_write> memo_sigx: array<f32>;

@group(1) @binding(13) // memory fields
                      // memory_dsigmaxy_dx, memory_dsigmayy_dy, memory_dsigmayz_dz
var<storage,read_write> memo_sigy: array<f32>;

@group(1) @binding(14) // memory fields
                      // memory_dsigmaxz_dx, memory_dsigmayz_dy, memory_dsigmazz_dz
var<storage,read_write> memo_sigz: array<f32>;

@group(1) @binding(15) // velocity ^ 2 (v_2)
var<storage,read_write> v_2: array<f32>;

// Group 2 - sensors arrays
@group(2) @binding(16) // sensors signals vx
var<storage,read_write> sensors_vx: array<f32>;

@group(2) @binding(17) // sensors signals vy
var<storage,read_write> sensors_vy: array<f32>;

@group(2) @binding(18) // sensors signals vz
var<storage,read_write> sensors_vz: array<f32>;

@group(2) @binding(19) // sensors positions
var<storage,read> sensors_pos_x: array<i32>;

@group(2) @binding(20) // sensors positions
var<storage,read> sensors_pos_y: array<i32>;

@group(2) @binding(21) // sensors positions
var<storage,read> sensors_pos_z: array<i32>;

// -------------------------------
// --- Index access functions ----
// -------------------------------
// function to convert 2D [i,j] index into 1D [] index
fn ij(i: i32, j: i32, i_max: i32, j_max: i32) -> i32 {
    let index = j + i * j_max;

    return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max);
}

// function to convert 4D [i,j,k,l] index into 1D [] index
fn ijkl(i: i32, j: i32, k: i32, l: i32, i_max: i32, j_max: i32, k_max: i32, l_max: i32) -> i32 {
    let index = k + j * k_max + i * k_max * j_max + l * k_max * j_max * i_max;

    return select(-1, index, i >= 0 && i < i_max &&
                             j >= 0 && j < j_max &&
                             k >= 0 && k < k_max &&
                             l >= 0 && l < l_max);
}

// ------------------------------------
// --- Force array access funtions ---
// ------------------------------------
// function to get a source_term array value
fn get_source_term(n: i32, e: i32) -> f32 {
    let index: i32 = ij(n, e, sim_int_par.n_iter, sim_int_par.n_probe_elem);

    return select(0.0, source_term[index], index != -1);
}

// function to get a source term index of a source
fn get_idx_source_term(x: i32, y: i32, z: i32) -> i32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 1);

    return select(-1, idx_src[index], index != -1);
}

// -------------------------------------------------
// --- CPML X coefficients array access funtions ---
// -------------------------------------------------
// function to get a a_x array value
fn get_a_x(n: i32) -> f32 {
    let index: i32 = ij(n, 0, sim_int_par.x_sz - 2, 6);

    return select(0.0, coef_x[index], index != -1);
}

// function to get a a_x_h array value
fn get_a_x_h(n: i32) -> f32 {
    let index: i32 = ij(n, 3, sim_int_par.x_sz - 2, 6);

    return select(0.0, coef_x[index], index != -1);
}

// function to get a b_x array value
fn get_b_x(n: i32) -> f32 {
    let index: i32 = ij(n, 1, sim_int_par.x_sz - 2, 6);

    return select(0.0, coef_x[index], index != -1);
}

// function to get a b_x_h array value
fn get_b_x_h(n: i32) -> f32 {
    let index: i32 = ij(n, 4, sim_int_par.x_sz - 2, 6);

    return select(0.0, coef_x[index], index != -1);
}

// function to get a k_x array value
fn get_k_x(n: i32) -> f32 {
    let index: i32 = ij(n, 2, sim_int_par.x_sz - 2, 6);

    return select(0.0, coef_x[index], index != -1);
}

// function to get a k_x_h array value
fn get_k_x_h(n: i32) -> f32 {
    let index: i32 = ij(n, 5, sim_int_par.x_sz - 2, 6);

    return select(0.0, coef_x[index], index != -1);
}

// -------------------------------------------------
// --- CPML Y coefficients array access funtions ---
// -------------------------------------------------
// function to get a a_y array value
fn get_a_y(n: i32) -> f32 {
    let index: i32 = ij(n, 0, sim_int_par.y_sz - 2, 6);

    return select(0.0, coef_y[index], index != -1);
}

// function to get a a_y_h array value
fn get_a_y_h(n: i32) -> f32 {
    let index: i32 = ij(n, 3, sim_int_par.y_sz - 2, 6);

    return select(0.0, coef_y[index], index != -1);
}

// function to get a b_y array value
fn get_b_y(n: i32) -> f32 {
    let index: i32 = ij(n, 1, sim_int_par.y_sz - 2, 6);

    return select(0.0, coef_y[index], index != -1);
}

// function to get a b_y_h array value
fn get_b_y_h(n: i32) -> f32 {
    let index: i32 = ij(n, 4, sim_int_par.y_sz - 2, 6);

    return select(0.0, coef_y[index], index != -1);
}

// function to get a k_y array value
fn get_k_y(n: i32) -> f32 {
    let index: i32 = ij(n, 2, sim_int_par.y_sz - 2, 6);

    return select(0.0, coef_y[index], index != -1);
}

// function to get a k_y_h array value
fn get_k_y_h(n: i32) -> f32 {
    let index: i32 = ij(n, 5, sim_int_par.y_sz - 2, 6);

    return select(0.0, coef_y[index], index != -1);
}

// -------------------------------------------------
// --- CPML Z coefficients array access funtions ---
// -------------------------------------------------
// function to get a a_z array value
fn get_a_z(n: i32) -> f32 {
    let index: i32 = ij(n, 0, sim_int_par.z_sz - 2, 6);

    return select(0.0, coef_z[index], index != -1);
}

// function to get a a_z_h array value
fn get_a_z_h(n: i32) -> f32 {
    let index: i32 = ij(n, 3, sim_int_par.z_sz - 2, 6);

    return select(0.0, coef_z[index], index != -1);
}

// function to get a b_z array value
fn get_b_z(n: i32) -> f32 {
    let index: i32 = ij(n, 1, sim_int_par.z_sz - 2, 6);

    return select(0.0, coef_z[index], index != -1);
}

// function to get a b_z_h array value
fn get_b_z_h(n: i32) -> f32 {
    let index: i32 = ij(n, 4, sim_int_par.z_sz - 2, 6);

    return select(0.0, coef_z[index], index != -1);
}

// function to get a k_z array value
fn get_k_z(n: i32) -> f32 {
    let index: i32 = ij(n, 2, sim_int_par.z_sz - 2, 6);

    return select(0.0, coef_z[index], index != -1);
}

// function to get a k_z_h array value
fn get_k_z_h(n: i32) -> f32 {
    let index: i32 = ij(n, 5, sim_int_par.z_sz - 2, 6);

    return select(0.0, coef_z[index], index != -1);
}

// ---------------------------------------
// --- Velocity arrays access funtions ---
// ---------------------------------------
// function to get a vx array value
fn get_vx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 4);

    return select(0.0, vel[index], index != -1);
}

// function to set a vx array value
fn set_vx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 4);

    if(index != -1) {
        vel[index] = val;
    }
}

// function to get a vy array value
fn get_vy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 4);

    return select(0.0, vel[index], index != -1);
}

// function to set a vy array value
fn set_vy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 4);

    if(index != -1) {
        vel[index] = val;
    }
}

// function to get a vz array value
fn get_vz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 4);

    return select(0.0, vel[index], index != -1);
}

// function to set a vz array value
fn set_vz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 4);

    if(index != -1) {
        vel[index] = val;
    }
}

// function to get a v_2 array value
fn get_v_2(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 1);

    return select(0.0, v_2[index], index != -1);
}

// function to set a v_2 array value
fn set_v_2(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 1);

    if(index != -1) {
        v_2[index] = val;
    }
}

// -------------------------------------
// --- Stress arrays access funtions ---
// -------------------------------------
// function to get a sigmaxx array value
fn get_sigmaxx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, sig_norm[index], index != -1);
}

// function to set a sigmaxx array value
fn set_sigmaxx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        sig_norm[index] = val;
    }
}

// function to get a sigmayy array value
fn get_sigmayy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, sig_norm[index], index != -1);
}

// function to set a sigmayy array value
fn set_sigmayy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        sig_norm[index] = val;
    }
}

// function to get a sigmazz array value
fn get_sigmazz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, sig_norm[index], index != -1);
}

// function to set a sigmazz array value
fn set_sigmazz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        sig_norm[index] = val;
    }
}

// function to get a sigmaxy array value
fn get_sigmaxy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, sig_trans[index], index != -1);
}

// function to set a sigmaxy array value
fn set_sigmaxy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        sig_trans[index] = val;
    }
}

// function to get a sigmaxz array value
fn get_sigmaxz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, sig_trans[index], index != -1);
}

// function to set a sigmaxz array value
fn set_sigmaxz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        sig_trans[index] = val;
    }
}

// function to get a sigmayz array value
fn get_sigmayz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, sig_trans[index], index != -1);
}

// function to set a sigmayz array value
fn set_sigmayz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        sig_trans[index] = val;
    }
}

// -------------------------------------
// --- Memory arrays access funtions ---
// -------------------------------------
// function to get a memory_dvx_dx array value
fn get_mdvx_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_v_dx[index], index != -1);
}

// function to set a memory_dvx_dx array value
fn set_mdvx_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_v_dx[index] = val;
    }
}

// function to get a memory_dvy_dx array value
fn get_mdvy_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_v_dx[index], index != -1);
}

// function to set a memory_dvy_dx array value
fn set_mdvy_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_v_dx[index] = val;
    }
}

// function to get a memory_dvz_dx array value
fn get_mdvz_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_v_dx[index], index != -1);
}

// function to set a memory_dvz_dx array value
fn set_mdvz_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_v_dx[index] = val;
    }
}

// function to get a memory_dvx_dy array value
fn get_mdvx_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_v_dy[index], index != -1);
}

// function to set a memory_dvx_dy array value
fn set_mdvx_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_v_dy[index] = val;
    }
}

// function to get a memory_dvy_dy array value
fn get_mdvy_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_v_dy[index], index != -1);
}

// function to set a memory_dvy_dy array value
fn set_mdvy_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_v_dy[index] = val;
    }
}

// function to get a memory_dvz_dy array value
fn get_mdvz_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_v_dy[index], index != -1);
}

// function to set a memory_dvz_dy array value
fn set_mdvz_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_v_dy[index] = val;
    }
}

// function to get a memory_dvx_dz array value
fn get_mdvx_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_v_dz[index], index != -1);
}

// function to set a memory_dvy_dx array value
fn set_mdvx_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_v_dz[index] = val;
    }
}

// function to get a memory_dvy_dz array value
fn get_mdvy_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_v_dz[index], index != -1);
}

// function to set a memory_dvy_dz array value
fn set_mdvy_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_v_dz[index] = val;
    }
}

// function to get a memory_dvz_dz array value
fn get_mdvz_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_v_dz[index], index != -1);
}

// function to set a memory_dvz_dz array value
fn set_mdvz_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_v_dz[index] = val;
    }
}

// function to get a memory_dsigmaxx_dx array value
fn get_mdsxx_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_sigx[index], index != -1);
}

// function to set a memory_dsigmaxx_dx array value
fn set_mdsxx_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_sigx[index] = val;
    }
}

// function to get a memory_dsigmaxy_dy array value
fn get_mdsxy_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_sigx[index], index != -1);
}

// function to set a memory_dsigmaxy_dy array value
fn set_mdsxy_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_sigx[index] = val;
    }
}

// function to get a memory_dsigmaxz_dz array value
fn get_mdsxz_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_sigx[index], index != -1);
}

// function to set a memory_dsigmaxz_dz array value
fn set_mdsxz_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_sigx[index] = val;
    }
}

// function to get a memory_dsigmaxy_dx array value
fn get_mdsxy_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_sigy[index], index != -1);
}

// function to set a memory_dsigmaxy_dx array value
fn set_mdsxy_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_sigy[index] = val;
    }
}

// function to get a memory_dsigmayy_dy array value
fn get_mdsyy_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_sigy[index], index != -1);
}

// function to set a memory_dsigmayy_dy array value
fn set_mdsyy_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_sigy[index] = val;
    }
}

// function to get a memory_dsigmayz_dz array value
fn get_mdsyz_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_sigy[index], index != -1);
}

// function to set a memory_dsigmayz_dz array value
fn set_mdsyz_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_sigy[index] = val;
    }
}

// function to get a memory_dsigmaxz_dx array value
fn get_mdsxz_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_sigz[index], index != -1);
}

// function to set a memory_dsigmaxz_dx array value
fn set_mdsxz_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_sigz[index] = val;
    }
}

// function to get a memory_dsigmayz_dy array value
fn get_mdsyz_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_sigz[index], index != -1);
}

// function to set a memory_dsigmayz_dy array value
fn set_mdsyz_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_sigz[index] = val;
    }
}

// function to get a memory_dsigmazz_dz array value
fn get_mdszz_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, memo_sigz[index], index != -1);
}

// function to set a memory_dsigmazz_dz array value
fn set_mdszz_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        memo_sigz[index] = val;
    }
}

// --------------------------------------
// --- Sensors arrays access funtions ---
// --------------------------------------
// function to set a sens_vx array value
fn set_sens_vx(n: i32, s: i32, val : f32) {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec);

    if(index != -1) {
        sensors_vx[index] = val;
    }
}

// function to set a sens_vy array value
fn set_sens_vy(n: i32, s: i32, val : f32) {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec);

    if(index != -1) {
        sensors_vy[index] = val;
    }
}

// function to set a sens_vz array value
fn set_sens_vz(n: i32, s: i32, val : f32) {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec);

    if(index != -1) {
        sensors_vz[index] = val;
    }
}

// function to get a x index position of a sensor
fn get_sens_pos_x(s: i32) -> i32 {
    return select(-1, sensors_pos_x[s], s >= 0 && s < sim_int_par.n_rec);
}

// function to get a y index position of a sensor
fn get_sens_pos_y(s: i32) -> i32 {
    return select(-1, sensors_pos_y[s], s >= 0 && s < sim_int_par.n_rec);
}

// function to get a z index position of a sensor
fn get_sens_pos_z(s: i32) -> i32 {
    return select(-1, sensors_pos_z[s], s >= 0 && s < sim_int_par.n_rec);
}

// -------------------------------------------------------------
// --- Finite difference index limits arrays access funtions ---
// -------------------------------------------------------------
// function to get an index to ini-half grid
fn get_idx_ih(c: i32) -> i32 {
    let index: i32 = ij(c, 0, sim_int_par.fd_coeff, 4);

    return select(-1, idx_fd[index], index != -1);
}

// function to get an index to ini-full grid
fn get_idx_if(c: i32) -> i32 {
    let index: i32 = ij(c, 1, sim_int_par.fd_coeff, 4);

    return select(-1, idx_fd[index], index != -1);
}

// function to get an index to fin-half grid
fn get_idx_fh(c: i32) -> i32 {
    let index: i32 = ij(c, 2, sim_int_par.fd_coeff, 4);

    return select(-1, idx_fd[index], index != -1);
}

// function to get an index to fin-full grid
fn get_idx_ff(c: i32) -> i32 {
    let index: i32 = ij(c, 3, sim_int_par.fd_coeff, 4);

    return select(-1, idx_fd[index], index != -1);
}

// function to get a fd coefficient
fn get_fdc(c: i32) -> f32 {
    return select(0.0, fd_coeffs[c], c >= 0 && c < sim_int_par.fd_coeff);
}

// ---------------
// --- Kernels ---
// ---------------
@compute
@workgroup_size(wsx, wsy, wsz)
fn teste_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let z: i32 = i32(index.z);          // z thread index

   let idx_src_term: i32 = get_idx_source_term(x, y, z);
   set_vz(x, y, z, f32(idx_src_term));
}

// Kernel to calculate stresses [sigmaxx, sigmayy, sigmaxy]
@compute
@workgroup_size(wsx, wsy, wsz)
fn sigma_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let z: i32 = i32(index.z);          // z thread index
    let dx: f32 = sim_flt_par.dx;
    let dy: f32 = sim_flt_par.dy;
    let dz: f32 = sim_flt_par.dz;
    let dt: f32 = sim_flt_par.dt;
    let lambda: f32 = sim_flt_par.lambda;
    let mu: f32 = sim_flt_par.mu;
    let lambdaplus2mu: f32 = lambda + 2.0 * mu;
    let last: i32 = sim_int_par.fd_coeff - 1;
    let offset: i32 = sim_int_par.fd_coeff - 1;

    // Normal stresses
    var id_x_i: i32 = -get_idx_fh(last);
    var id_x_f: i32 = sim_int_par.x_sz - get_idx_ih(last);
    var id_y_i: i32 = -get_idx_ff(last);
    var id_y_f: i32 = sim_int_par.y_sz - get_idx_if(last);
    var id_z_i: i32 = -get_idx_ff(last);
    var id_z_f: i32 = sim_int_par.z_sz - get_idx_if(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdvx_dx: f32 = 0.0;
        var vdvy_dy: f32 = 0.0;
        var vdvz_dz: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdvx_dx += get_fdc(c) * (get_vx(x + get_idx_ih(c), y, z) - get_vx(x + get_idx_fh(c), y, z)) / dx;
            vdvy_dy += get_fdc(c) * (get_vy(x, y + get_idx_if(c), z) - get_vy(x, y + get_idx_ff(c), z)) / dy;
            vdvz_dz += get_fdc(c) * (get_vz(x, y, z + get_idx_if(c)) - get_vz(x, y, z + get_idx_ff(c))) / dz;
        }

        var mdvx_dx_new: f32 = get_b_x_h(x - offset) * get_mdvx_dx(x, y, z) + get_a_x_h(x - offset) * vdvx_dx;
        var mdvy_dy_new: f32 = get_b_y(y - offset) * get_mdvy_dy(x, y, z) + get_a_y(y - offset) * vdvy_dy;
        var mdvz_dz_new: f32 = get_b_z(z - offset) * get_mdvz_dz(x, y, z) + get_a_z(z - offset) * vdvz_dz;

        vdvx_dx = vdvx_dx/get_k_x_h(x - offset) + mdvx_dx_new;
        vdvy_dy = vdvy_dy/get_k_y(y - offset)  + mdvy_dy_new;
        vdvz_dz = vdvz_dz/get_k_z(z - offset)  + mdvz_dz_new;

        set_mdvx_dx(x, y, z, mdvx_dx_new);
        set_mdvy_dy(x, y, z, mdvy_dy_new);
        set_mdvz_dz(x, y, z, mdvz_dz_new);

        set_sigmaxx(x, y, z, get_sigmaxx(x, y, z) + (lambdaplus2mu * vdvx_dx + lambda        * (vdvy_dy + vdvz_dz))*dt);
        set_sigmayy(x, y, z, get_sigmayy(x, y, z) + (lambda        * (vdvx_dx + vdvz_dz) + lambdaplus2mu * vdvy_dy)*dt);
        set_sigmazz(x, y, z, get_sigmazz(x, y, z) + (lambda        * (vdvx_dx + vdvy_dy) + lambdaplus2mu * vdvz_dz)*dt);
    }

    // Shear stresses
    // sigma_xy
    id_x_i = -get_idx_ff(last);
    id_x_f = sim_int_par.x_sz - get_idx_if(last);
    id_y_i = -get_idx_fh(last);
    id_y_f = sim_int_par.y_sz - get_idx_ih(last);
    id_z_i = -get_idx_fh(last);
    id_z_f = sim_int_par.z_sz - get_idx_ih(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdvy_dx: f32 = 0.0;
        var vdvx_dy: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdvy_dx += get_fdc(c) * (get_vy(x + get_idx_if(c), y, z) - get_vy(x + get_idx_ff(c), y, z)) / dx;
            vdvx_dy += get_fdc(c) * (get_vx(x, y + get_idx_ih(c), z) - get_vx(x, y + get_idx_fh(c), z)) / dy;
        }

        var mdvy_dx_new: f32 = get_b_x(x - offset) * get_mdvy_dx(x, y, z) + get_a_x(x - offset) * vdvy_dx;
        var mdvx_dy_new: f32 = get_b_y_h(y - offset) * get_mdvx_dy(x, y, z) + get_a_y_h(y - offset) * vdvx_dy;

        vdvy_dx = vdvy_dx/get_k_x(x - offset)   + mdvy_dx_new;
        vdvx_dy = vdvx_dy/get_k_y_h(y - offset) + mdvx_dy_new;

        set_mdvy_dx(x, y, z, mdvy_dx_new);
        set_mdvx_dy(x, y, z, mdvx_dy_new);

        set_sigmaxy(x, y, z, get_sigmaxy(x, y, z) + (vdvx_dy + vdvy_dx) * mu * dt);
    }

    // sigma_xz
    id_x_i = -get_idx_ff(last);
    id_x_f = sim_int_par.x_sz - get_idx_if(last);
    id_y_i = -get_idx_fh(last);
    id_y_f = sim_int_par.y_sz - get_idx_ih(last);
    id_z_i = -get_idx_fh(last);
    id_z_f = sim_int_par.z_sz - get_idx_ih(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdvz_dx: f32 = 0.0;
        var vdvx_dz: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdvz_dx += get_fdc(c) * (get_vz(x + get_idx_if(c), y, z) - get_vz(x + get_idx_ff(c), y, z)) / dx;
            vdvx_dz += get_fdc(c) * (get_vx(x, y, z + get_idx_ih(c)) - get_vx(x, y, z + get_idx_fh(c))) / dz;
        }

        var mdvz_dx_new: f32 = get_b_x(x - offset) * get_mdvz_dx(x, y, z) + get_a_x(x - offset) * vdvz_dx;
        var mdvx_dz_new: f32 = get_b_z_h(z - offset) * get_mdvx_dz(x, y, z) + get_a_z_h(z - offset) * vdvx_dz;

        vdvz_dx = vdvz_dx/get_k_x(x - offset)   + mdvz_dx_new;
        vdvx_dz = vdvx_dz/get_k_z_h(z - offset) + mdvx_dz_new;

        set_mdvz_dx(x, y, z, mdvz_dx_new);
        set_mdvx_dz(x, y, z, mdvx_dz_new);

        set_sigmaxz(x, y, z, get_sigmaxz(x, y, z) + (vdvx_dz + vdvz_dx) * mu * dt);
    }

    // sigma_yz
    id_x_i = -get_idx_fh(last);
    id_x_f = sim_int_par.x_sz - get_idx_ih(last);
    id_y_i = -get_idx_fh(last);
    id_y_f = sim_int_par.y_sz - get_idx_ih(last);
    id_z_i = -get_idx_fh(last);
    id_z_f = sim_int_par.z_sz - get_idx_ih(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdvz_dy: f32 = 0.0;
        var vdvy_dz: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdvz_dy += get_fdc(c) * (get_vz(x, y + get_idx_ih(c), z) - get_vz(x, y + get_idx_fh(c), z)) / dy;
            vdvy_dz += get_fdc(c) * (get_vy(x, y, z + get_idx_ih(c)) - get_vy(x, y, z + get_idx_fh(c))) / dz;
        }

        var mdvz_dy_new: f32 = get_b_y_h(y - offset) * get_mdvz_dy(x, y, z) + get_a_y_h(y - offset) * vdvz_dy;
        var mdvy_dz_new: f32 = get_b_z_h(z - offset) * get_mdvy_dz(x, y, z) + get_a_z_h(z - offset) * vdvy_dz;

        vdvz_dy = vdvz_dy/get_k_y_h(y - offset) + mdvz_dy_new;
        vdvy_dz = vdvy_dz/get_k_z_h(z - offset) + mdvy_dz_new;

        set_mdvz_dy(x, y, z, mdvz_dy_new);
        set_mdvy_dz(x, y, z, mdvy_dz_new);

        set_sigmayz(x, y, z, get_sigmayz(x, y, z) + (vdvy_dz + vdvz_dy) * mu * dt);
    }
}

// Kernel to calculate velocities [vx, vy, vz]
@compute
@workgroup_size(wsx, wsy, wsz)
fn velocity_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let z: i32 = i32(index.z);          // z thread index
    let dt_over_rho: f32 = sim_flt_par.dt / sim_flt_par.rho;
    let dx: f32 = sim_flt_par.dx;
    let dy: f32 = sim_flt_par.dy;
    let dz: f32 = sim_flt_par.dz;
    let last: i32 = sim_int_par.fd_coeff - 1;
    let offset: i32 = sim_int_par.fd_coeff - 1;

    // Vx
    var id_x_i: i32 = -get_idx_ff(last);
    var id_x_f: i32 = sim_int_par.x_sz - get_idx_if(last);
    var id_y_i: i32 = -get_idx_ff(last);
    var id_y_f: i32 = sim_int_par.y_sz - get_idx_if(last);
    var id_z_i: i32 = -get_idx_ff(last);
    var id_z_f: i32 = sim_int_par.z_sz - get_idx_if(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdsigmaxx_dx: f32 = 0.0;
        var vdsigmaxy_dy: f32 = 0.0;
        var vdsigmaxz_dz: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdsigmaxx_dx += get_fdc(c) * (get_sigmaxx(x + get_idx_if(c), y, z) - get_sigmaxx(x + get_idx_ff(c), y, z)) / dx;
            vdsigmaxy_dy += get_fdc(c) * (get_sigmaxy(x, y + get_idx_if(c), z) - get_sigmaxy(x, y + get_idx_ff(c), z)) / dy;
            vdsigmaxz_dz += get_fdc(c) * (get_sigmaxz(x, y, z + get_idx_if(c)) - get_sigmaxz(x, y, z + get_idx_ff(c))) / dz;
        }

        var mdsxx_dx_new: f32 = get_b_x(x - offset) * get_mdsxx_dx(x, y, z) + get_a_x(x - offset) * vdsigmaxx_dx;
        var mdsxy_dy_new: f32 = get_b_y(y - offset) * get_mdsxy_dy(x, y, z) + get_a_y(y - offset) * vdsigmaxy_dy;
        var mdsxz_dz_new: f32 = get_b_z(z - offset) * get_mdsxz_dz(x, y, z) + get_a_z(z - offset) * vdsigmaxz_dz;

        vdsigmaxx_dx = vdsigmaxx_dx/get_k_x(x - offset) + mdsxx_dx_new;
        vdsigmaxy_dy = vdsigmaxy_dy/get_k_y(y - offset) + mdsxy_dy_new;
        vdsigmaxz_dz = vdsigmaxz_dz/get_k_z(z - offset) + mdsxz_dz_new;

        set_mdsxx_dx(x, y, z, mdsxx_dx_new);
        set_mdsxy_dy(x, y, z, mdsxy_dy_new);
        set_mdsxz_dz(x, y, z, mdsxz_dz_new);

        set_vx(x, y, z, dt_over_rho * (vdsigmaxx_dx + vdsigmaxy_dy + vdsigmaxz_dz) + get_vx(x, y, z));
    }

    // Vy
    id_x_i = -get_idx_fh(last);
    id_x_f = sim_int_par.x_sz - get_idx_ih(last);
    id_y_i = -get_idx_fh(last);
    id_y_f = sim_int_par.y_sz - get_idx_ih(last);
    id_z_i = -get_idx_ff(last);
    id_z_f = sim_int_par.z_sz - get_idx_if(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdsigmaxy_dx: f32 = 0.0;
        var vdsigmayy_dy: f32 = 0.0;
        var vdsigmayz_dz: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdsigmaxy_dx += get_fdc(c) * (get_sigmaxy(x + get_idx_ih(c), y, z) - get_sigmaxy(x + get_idx_fh(c), y, z)) / dx;
            vdsigmayy_dy += get_fdc(c) * (get_sigmayy(x, y + get_idx_ih(c), z) - get_sigmayy(x, y + get_idx_fh(c), z)) / dy;
            vdsigmayz_dz += get_fdc(c) * (get_sigmayz(x, y, z + get_idx_if(c)) - get_sigmayz(x, y, z + get_idx_ff(c))) / dz;
        }

        var mdsxy_dx_new: f32 = get_b_x_h(x - offset) * get_mdsxy_dx(x, y, z) + get_a_x_h(x - offset) * vdsigmaxy_dx;
        var mdsyy_dy_new: f32 = get_b_y_h(y - offset) * get_mdsyy_dy(x, y, z) + get_a_y_h(y - offset) * vdsigmayy_dy;
        var mdsyz_dz_new: f32 = get_b_z(z - offset)   * get_mdsyz_dz(x, y, z) + get_a_z(z - offset)   * vdsigmayz_dz;

        vdsigmaxy_dx = vdsigmaxy_dx/get_k_x_h(x - offset) + mdsxy_dx_new;
        vdsigmayy_dy = vdsigmayy_dy/get_k_y_h(y - offset) + mdsyy_dy_new;
        vdsigmayz_dz = vdsigmayz_dz/get_k_z(z - offset)   + mdsyz_dz_new;

        set_mdsxy_dx(x, y, z, mdsxy_dx_new);
        set_mdsyy_dy(x, y, z, mdsyy_dy_new);
        set_mdsyz_dz(x, y, z, mdsyz_dz_new);

        set_vy(x, y, z, dt_over_rho * (vdsigmaxy_dx + vdsigmayy_dy + vdsigmayz_dz) + get_vy(x, y, z));
    }

    // Vz
    id_x_i = -get_idx_fh(last);
    id_x_f = sim_int_par.x_sz - get_idx_ih(last);
    id_y_i = -get_idx_ff(last);
    id_y_f = sim_int_par.y_sz - get_idx_if(last);
    id_z_i = -get_idx_fh(last);
    id_z_f = sim_int_par.z_sz - get_idx_ih(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f && z >= id_z_i && z < id_z_f) {
        var vdsigmaxz_dx: f32 = 0.0;
        var vdsigmayz_dy: f32 = 0.0;
        var vdsigmazz_dz: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdsigmaxz_dx += get_fdc(c) * (get_sigmaxz(x + get_idx_ih(c), y, z) - get_sigmaxz(x + get_idx_fh(c), y, z)) / dx;
            vdsigmayz_dy += get_fdc(c) * (get_sigmayz(x, y + get_idx_if(c), z) - get_sigmayz(x, y + get_idx_ff(c), z)) / dy;
            vdsigmazz_dz += get_fdc(c) * (get_sigmazz(x, y, z + get_idx_ih(c)) - get_sigmazz(x, y, z + get_idx_fh(c))) / dz;
        }

        var mdsxz_dx_new: f32 = get_b_x_h(x - offset) * get_mdsxz_dx(x, y, z) + get_a_x_h(x - offset) * vdsigmaxz_dx;
        var mdsyz_dy_new: f32 = get_b_y(y - offset)   * get_mdsyz_dy(x, y, z) + get_a_y(y - offset)   * vdsigmayz_dy;
        var mdszz_dz_new: f32 = get_b_z_h(z - offset) * get_mdszz_dz(x, y, z) + get_a_z_h(z - offset) * vdsigmazz_dz;

        vdsigmaxz_dx = vdsigmaxz_dx/get_k_x_h(x - offset) + mdsxz_dx_new;
        vdsigmayz_dy = vdsigmayz_dy/get_k_y(y - offset)   + mdsyz_dy_new;
        vdsigmazz_dz = vdsigmazz_dz/get_k_z_h(z - offset) + mdszz_dz_new;

        set_mdsxz_dx(x, y, z, mdsxz_dx_new);
        set_mdsyz_dy(x, y, z, mdsyz_dy_new);
        set_mdszz_dz(x, y, z, mdszz_dz_new);

        set_vz(x, y, z, dt_over_rho * (vdsigmaxz_dx + vdsigmayz_dy + vdsigmazz_dz) + get_vz(x, y, z));
    }
}

// Kernel to add the sources forces
@compute
@workgroup_size(wsx, wsy, wsz)
fn sources_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let z: i32 = i32(index.z);          // y thread index
    let dt_over_rho: f32 = sim_flt_par.dt / sim_flt_par.rho;
    let it: i32 = sim_int_par.it;

    // Add the source force
    let idx_src_term: i32 = get_idx_source_term(x, y, z);
    if(idx_src_term != -1) {
        set_vz(x, y, z, get_vz(x, y, z) + get_source_term(it, idx_src_term) * dt_over_rho);
    }
}

// Kernel to finish iteration term
@compute
@workgroup_size(wsx, wsy, wsz)
fn finish_it_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let z: i32 = i32(index.z);          // y thread index
    let it: i32 = sim_int_par.it;
    let last: i32 = sim_int_par.fd_coeff - 1;
    let id_x_i: i32 = -get_idx_fh(last);
    let id_x_f: i32 = sim_int_par.x_sz - get_idx_ih(last);
    let id_y_i: i32 = -get_idx_fh(last);
    let id_y_f: i32 = sim_int_par.y_sz - get_idx_ih(last);
    let id_z_i: i32 = -get_idx_fh(last);
    let id_z_f: i32 = sim_int_par.z_sz - get_idx_ih(last);


    // Apply Dirichlet conditions
    if(x <= id_x_i || x >= id_x_f || y <= id_y_i || y >= id_y_f || z <= id_z_i || z >= id_z_f) {
        set_vx(x, y, z, 0.0);
        set_vy(x, y, z, 0.0);
        set_vz(x, y, z, 0.0);
    }

    // Store sensor velocities
    for(var s: i32 = 0; s < sim_int_par.n_rec; s++) {
        if(x == get_sens_pos_x(s) && y == get_sens_pos_y(s) && z == get_sens_pos_z(s)) {
            set_sens_vx(it, s, get_vx(x, y, z));
            set_sens_vy(it, s, get_vy(x, y, z));
            set_sens_vz(it, s, get_vz(x, y, z));
            break;
        }
    }

    // Compute velocity norm L2
    set_v_2(x, y, z, get_vx(x, y, z)*get_vx(x, y, z) +
                     get_vy(x, y, z)*get_vy(x, y, z) +
                     get_vz(x, y, z)*get_vz(x, y, z));
}

// Kernel to increase time iteraction [it]
@compute
@workgroup_size(1)
fn incr_it_kernel() {
    sim_int_par.it += 1;
}
