struct SimIntValues {
    x_sz: i32,          // x field size
    y_sz: i32,          // y field size
    n_iter: i32,        // num iterations
    n_src: i32,         // num sources
    n_rec: i32,         // num receptors
    fd_coeff: i32,      // num fd coefficients
    it: i32             // time iteraction
};

struct SimFltValues {
    cp: f32,            // longitudinal sound speed
    cs: f32,            // transverse sound speed
    dx: f32,            // delta x
    dy: f32,            // delta y
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

@group(0) @binding(22) // source positions
var<storage,read> sources_pos_x: array<i32>;

@group(0) @binding(23) // source positions
var<storage,read> sources_pos_y: array<i32>;

@group(0) @binding(2) // a_x, b_x, k_x, a_x_h, b_x_h, k_x_h
var<storage,read> coef_x: array<f32>;

@group(0) @binding(3) // a_y, b_y, k_y, a_y_h, b_y_h, k_y_h
var<storage,read> coef_y: array<f32>;

@group(0) @binding(4) // param_int32
var<storage,read_write> sim_int_par: SimIntValues;

@group(0) @binding(25) // idx_fd
var<storage,read> idx_fd: array<i32>;

@group(0) @binding(28) // fd_coeff
var<storage,read> fd_coeffs: array<f32>;

// Group 1 - simulation arrays
@group(1) @binding(5) // velocity fields (vx, vy, v_2)
var<storage,read_write> vel: array<f32>;

@group(1) @binding(6) // stress fields (sigmaxx, sigmayy, sigmaxy)
var<storage,read_write> sig: array<f32>;

@group(1) @binding(7) // memory fields
                      // memory_dvx_dx, memory_dvx_dy, memory_dvy_dx, memory_dvy_dy,
                      // memory_dsigmaxx_dx, memory_dsigmayy_dy, memory_dsigmaxy_dx, memory_dsigmaxy_dy
var<storage,read_write> memo: array<f32>;

// Group 2 - sensors arrays and energies
@group(2) @binding(8) // sensors signals vx
var<storage,read_write> sensors_vx: array<f32>;

@group(2) @binding(9) // sensors signals vy
var<storage,read_write> sensors_vy: array<f32>;

@group(2) @binding(10) // sensors positions
var<storage,read> sensors_pos_x: array<i32>;

@group(2) @binding(11) // sensors positions
var<storage,read> sensors_pos_y: array<i32>;

// -------------------------------
// --- Index access functions ----
// -------------------------------
// function to convert 2D [i,j] index into 1D [] index
fn ij(i: i32, j: i32, i_max: i32, j_max: i32) -> i32 {
    let index = j + i * j_max;

    return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max);
}

// function to convert 3D [i,j,k] index into 1D [] index
fn ijk(i: i32, j: i32, k: i32, i_max: i32, j_max: i32, k_max: i32) -> i32 {
    let index = j + i * j_max + k * j_max * i_max;

    return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max);
}

// ------------------------------------
// --- Force array access funtions ---
// ------------------------------------
// function to get a source_term array value
fn get_source_term(n: i32, s: i32) -> f32 {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_src);

    return select(0.0, source_term[index], index != -1);
}

// function to get a x index position of a source
fn get_sour_pos_x(s: i32) -> i32 {
    return select(-1, sources_pos_x[s], s >= 0 && s < sim_int_par.n_src);
}

// function to get a y index position of a source
fn get_sour_pos_y(s: i32) -> i32 {
    return select(-1, sources_pos_y[s], s >= 0 && s < sim_int_par.n_src);
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

// ---------------------------------------
// --- Velocity arrays access funtions ---
// ---------------------------------------
// function to get a vx array value
fn get_vx(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 3);

    return select(0.0, vel[index], index != -1);
}

// function to set a vx array value
fn set_vx(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 3);

    if(index != -1) {
        vel[index] = val;
    }
}

// function to get a vy array value
fn get_vy(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 3);

    return select(0.0, vel[index], index != -1);
}

// function to set a vy array value
fn set_vy(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 3);

    if(index != -1) {
        vel[index] = val;
    }
}

// function to get a v_2 array value
fn get_v_2(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 3);

    return select(0.0, vel[index], index != -1);
}

// function to set a v_2 array value
fn set_v_2(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 3);

    if(index != -1) {
        vel[index] = val;
    }
}

// -------------------------------------
// --- Stress arrays access funtions ---
// -------------------------------------
// function to get a sigmaxx array value
fn get_sigmaxx(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 3);

    return select(0.0, sig[index], index != -1);
}

// function to set a sigmaxx array value
fn set_sigmaxx(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 3);

    if(index != -1) {
        sig[index] = val;
    }
}

// function to get a sigmayy array value
fn get_sigmayy(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 3);

    return select(0.0, sig[index], index != -1);
}

// function to set a sigmayy array value
fn set_sigmayy(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 3);

    if(index != -1) {
        sig[index] = val;
    }
}

// function to get a sigmaxy array value
fn get_sigmaxy(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 3);

    return select(0.0, sig[index], index != -1);
}

// function to set a sigmaxy array value
fn set_sigmaxy(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 3);

    if(index != -1) {
        sig[index] = val;
    }
}

// -------------------------------------
// --- Memory arrays access funtions ---
// -------------------------------------
// function to get a memory_dvx_dx array value
fn get_mdvx_dx(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvx_dx array value
fn set_mdvx_dx(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dvx_dy array value
fn get_mdvx_dy(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvx_dy array value
fn set_mdvx_dy(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dvy_dx array value
fn get_mdvy_dx(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvy_dx array value
fn set_mdvy_dx(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dvy_dy array value
fn get_mdvy_dy(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvy_dy array value
fn set_mdvy_dy(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmaxx_dx array value
fn get_mdsxx_dx(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 4, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmaxx_dx array value
fn set_mdsxx_dx(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 4, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmayy_dy array value
fn get_mdsyy_dy(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 5, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmayy_dy array value
fn set_mdsyy_dy(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 5, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmaxy_dx array value
fn get_mdsxy_dx(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 6, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmaxy_dx array value
fn set_mdsxy_dx(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 6, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmaxy_dy array value
fn get_mdsxy_dy(x: i32, y: i32) -> f32 {
    let index: i32 = ijk(x, y, 7, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmaxy_dy array value
fn set_mdsxy_dy(x: i32, y: i32, val : f32) {
    let index: i32 = ijk(x, y, 7, sim_int_par.x_sz, sim_int_par.y_sz, 8);

    if(index != -1) {
        memo[index] = val;
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

// function to get a x index position of a sensor
fn get_sens_pos_x(s: i32) -> i32 {
    return select(-1, sensors_pos_x[s], s >= 0 && s < sim_int_par.n_rec);
}

// function to get a y index position of a sensor
fn get_sens_pos_y(s: i32) -> i32 {
    return select(-1, sensors_pos_y[s], s >= 0 && s < sim_int_par.n_rec);
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
@workgroup_size(wsx, wsy)
fn teste_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let dx: f32 = sim_flt_par.dx;
    let dy: f32 = sim_flt_par.dy;
    let dt: f32 = sim_flt_par.dt;
    let lambda: f32 = sim_flt_par.lambda;
    let mu: f32 = sim_flt_par.mu;
    let lambdaplus2mu: f32 = lambda + 2.0 * mu;
    let last: i32 = sim_int_par.fd_coeff - 1;
    let offset_x: i32 = sim_int_par.fd_coeff - 1;
    let offset_y: i32 = sim_int_par.fd_coeff - 1;

    // Normal stresses
    var id_x_i: i32 = -get_idx_fh(last);
    var id_x_f: i32 = sim_int_par.x_sz - get_idx_ih(last);
    var id_y_i: i32 = -get_idx_ff(last);
    var id_y_f: i32 = sim_int_par.y_sz - get_idx_if(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
        set_vx(x, y, get_b_x_h(x - offset_x));
        set_vy(x, y, get_b_y(y - offset_y));
    }
}

// Kernel to calculate stresses [sigmaxx, sigmayy, sigmaxy]
@compute
@workgroup_size(wsx, wsy)
fn sigma_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let dx: f32 = sim_flt_par.dx;
    let dy: f32 = sim_flt_par.dy;
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
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
        var vdvx_dx: f32 = 0.0;
        var vdvy_dy: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdvx_dx += get_fdc(c) * (get_vx(x + get_idx_ih(c), y) - get_vx(x + get_idx_fh(c), y)) / dx;
            vdvy_dy += get_fdc(c) * (get_vy(x, y + get_idx_if(c)) - get_vy(x, y + get_idx_ff(c))) / dy;
        }

        var mdvx_dx_new: f32 = get_b_x_h(x - offset) * get_mdvx_dx(x, y) + get_a_x_h(x - offset) * vdvx_dx;
        var mdvy_dy_new: f32 = get_b_y(y - offset) * get_mdvy_dy(x, y) + get_a_y(y - offset) * vdvy_dy;

        vdvx_dx = vdvx_dx/get_k_x_h(x - offset) + mdvx_dx_new;
        vdvy_dy = vdvy_dy/get_k_y(y - offset)  + mdvy_dy_new;

        set_mdvx_dx(x, y, mdvx_dx_new);
        set_mdvy_dy(x, y, mdvy_dy_new);

        set_sigmaxx(x, y, get_sigmaxx(x, y) + (lambdaplus2mu * vdvx_dx + lambda        * vdvy_dy)*dt);
        set_sigmayy(x, y, get_sigmayy(x, y) + (lambda        * vdvx_dx + lambdaplus2mu * vdvy_dy)*dt);
    }

    // Shear stresses
    // sigma_xy
    id_x_i = -get_idx_ff(last);
    id_x_f = sim_int_par.x_sz - get_idx_if(last);
    id_y_i = -get_idx_fh(last);
    id_y_f = sim_int_par.y_sz - get_idx_ih(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
        var vdvy_dx: f32 = 0.0;
        var vdvx_dy: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdvy_dx += get_fdc(c) * (get_vy(x + get_idx_if(c), y) - get_vy(x + get_idx_ff(c), y)) / dx;
            vdvx_dy += get_fdc(c) * (get_vx(x, y + get_idx_ih(c)) - get_vx(x, y + get_idx_fh(c))) / dy;
        }

        var mdvy_dx_new: f32 = get_b_x(x - offset) * get_mdvy_dx(x, y) + get_a_x(x - offset) * vdvy_dx;
        var mdvx_dy_new: f32 = get_b_y_h(y - offset) * get_mdvx_dy(x, y) + get_a_y_h(y - offset) * vdvx_dy;

        vdvy_dx = vdvy_dx/get_k_x(x - offset)   + mdvy_dx_new;
        vdvx_dy = vdvx_dy/get_k_y_h(y - offset) + mdvx_dy_new;

        set_mdvy_dx(x, y, mdvy_dx_new);
        set_mdvx_dy(x, y, mdvx_dy_new);

        set_sigmaxy(x, y, get_sigmaxy(x, y) + (vdvx_dy + vdvy_dx) * mu * dt);
    }
}

// Kernel to calculate velocities [vx, vy]
@compute
@workgroup_size(wsx, wsy)
fn velocity_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let dt_over_rho: f32 = sim_flt_par.dt / sim_flt_par.rho;
    let dx: f32 = sim_flt_par.dx;
    let dy: f32 = sim_flt_par.dy;
    let last: i32 = sim_int_par.fd_coeff - 1;
    let offset: i32 = sim_int_par.fd_coeff - 1;

    // Vx
    var id_x_i: i32 = -get_idx_ff(last);
    var id_x_f: i32 = sim_int_par.x_sz - get_idx_if(last);
    var id_y_i: i32 = -get_idx_ff(last);
    var id_y_f: i32 = sim_int_par.y_sz - get_idx_if(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
        var vdsigmaxx_dx: f32 = 0.0;
        var vdsigmaxy_dy: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdsigmaxx_dx += get_fdc(c) * (get_sigmaxx(x + get_idx_if(c), y) - get_sigmaxx(x + get_idx_ff(c), y)) / dx;
            vdsigmaxy_dy += get_fdc(c) * (get_sigmaxy(x, y + get_idx_if(c)) - get_sigmaxy(x, y + get_idx_ff(c))) / dy;
        }

        var mdsxx_dx_new: f32 = get_b_x(x - offset) * get_mdsxx_dx(x, y) + get_a_x(x - offset) * vdsigmaxx_dx;
        var mdsxy_dy_new: f32 = get_b_y(y - offset) * get_mdsxy_dy(x, y) + get_a_y(y - offset) * vdsigmaxy_dy;

        vdsigmaxx_dx = vdsigmaxx_dx/get_k_x(x - offset) + mdsxx_dx_new;
        vdsigmaxy_dy = vdsigmaxy_dy/get_k_y(y - offset) + mdsxy_dy_new;

        set_mdsxx_dx(x, y, mdsxx_dx_new);
        set_mdsxy_dy(x, y, mdsxy_dy_new);

        set_vx(x, y, dt_over_rho * (vdsigmaxx_dx + vdsigmaxy_dy) + get_vx(x, y));
    }

    // Vy
    id_x_i = -get_idx_fh(last);
    id_x_f = sim_int_par.x_sz - get_idx_ih(last);
    id_y_i = -get_idx_fh(last);
    id_y_f = sim_int_par.y_sz - get_idx_ih(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
        var vdsigmaxy_dx: f32 = 0.0;
        var vdsigmayy_dy: f32 = 0.0;
        for(var c: i32 = 0; c < sim_int_par.fd_coeff; c++) {
            vdsigmaxy_dx += get_fdc(c) * (get_sigmaxy(x + get_idx_ih(c), y) - get_sigmaxy(x + get_idx_fh(c), y)) / dx;
            vdsigmayy_dy += get_fdc(c) * (get_sigmayy(x, y + get_idx_ih(c)) - get_sigmayy(x, y + get_idx_fh(c))) / dy;
        }

        var mdsxy_dx_new: f32 = get_b_x_h(x - offset) * get_mdsxy_dx(x, y) + get_a_x_h(x - offset) * vdsigmaxy_dx;
        var mdsyy_dy_new: f32 = get_b_y_h(y - offset) * get_mdsyy_dy(x, y) + get_a_y_h(y - offset) * vdsigmayy_dy;

        vdsigmaxy_dx = vdsigmaxy_dx/get_k_x_h(x - offset) + mdsxy_dx_new;
        vdsigmayy_dy = vdsigmayy_dy/get_k_y_h(y - offset) + mdsyy_dy_new;

        set_mdsxy_dx(x, y, mdsxy_dx_new);
        set_mdsyy_dy(x, y, mdsyy_dy_new);

        set_vy(x, y, dt_over_rho * (vdsigmaxy_dx + vdsigmayy_dy) + get_vy(x, y));
    }
}

// Kernel to finish iteration term
@compute
@workgroup_size(wsx, wsy)
fn finish_it_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let dt_over_rho: f32 = sim_flt_par.dt / sim_flt_par.rho;
    let it: i32 = sim_int_par.it;
    let last: i32 = sim_int_par.fd_coeff - 1;
    let id_x_i: i32 = -get_idx_fh(last);
    let id_x_f: i32 = sim_int_par.x_sz - get_idx_ih(last);
    let id_y_i: i32 = -get_idx_fh(last);
    let id_y_f: i32 = sim_int_par.y_sz - get_idx_ih(last);

    // Add the source force
    for(var s: i32 = 0; s < sim_int_par.n_src; s++) {
        if(x == get_sour_pos_x(s) && y == get_sour_pos_y(s)) {
            set_vy(x, y, get_vy(x, y) + get_source_term(it, s) * dt_over_rho);
            break;
        }
    }

    // Apply Dirichlet conditions
    if(x <= id_x_i || x >= id_x_f || y <= id_y_i || y >= id_y_f) {
        set_vx(x, y, 0.0);
        set_vy(x, y, 0.0);
    }

    // Store sensors velocities
    for(var s: i32 = 0; s < sim_int_par.n_rec; s++) {
        if(x == get_sens_pos_x(s) && y == get_sens_pos_y(s)) {
            set_sens_vx(it, s, get_vx(x, y));
            set_sens_vy(it, s, get_vy(x, y));
            break;
        }
    }

    // Compute velocity norm L2
    set_v_2(x, y, get_vx(x, y)*get_vx(x, y) + get_vy(x, y)*get_vy(x, y));
}

// Kernel to increase time iteraction [it]
@compute
@workgroup_size(1)
fn incr_it_kernel() {
    sim_int_par.it += 1;
}
