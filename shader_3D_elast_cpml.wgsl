struct SimIntValues {
    x_sz: i32,          // x field size
    y_sz: i32,          // y field size
    z_sz: i32,          // z field size
    x_source: i32,      // x source index
    y_source: i32,      // y source index
    z_source: i32,      // z source index
    x_sens: i32,        // x sensor
    y_sens: i32,        // y sensor
    z_sens: i32,        // z sensor
    np_pml: i32,        // PML size
    n_iter: i32,        // max iterations
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

@group(0) @binding(1) // force terms
var<storage,read> force: array<f32>;

@group(0) @binding(2) // a_x, b_x, k_x, a_x_h, b_x_h, k_x_h
var<storage,read> coef_x: array<f32>;

@group(0) @binding(3) // a_y, b_y, k_y, a_y_h, b_y_h, k_y_h
var<storage,read> coef_y: array<f32>;

@group(0) @binding(4) // a_z, b_z, k_z, a_z_h, b_z_h, k_z_h
var<storage,read> coef_z: array<f32>;

@group(0) @binding(5) // param_int32
var<storage,read_write> sim_int_par: SimIntValues;

// Group 1 - simulation arrays
@group(1) @binding(6) // velocity fields (vx, vy, vz)
var<storage,read_write> vel: array<f32>;

@group(1) @binding(7) // stress fields (sigmaxx, sigmayy, sigmazz, sigmaxy, sigmaxz, sigmayz)
var<storage,read_write> sig: array<f32>;

@group(1) @binding(8) // memory fields
                      // memory_dvx_dx, memory_dvx_dy, memory_dvx_dz
                      // memory_dvx_dy, memory_dvy_dy, memory_dvz_dy
                      // memory_dvx_dz, memory_dvy_dz, memory_dvz_dz
                      // memory_dsigmaxx_dx, memory_dsigmaxy_dy, memory_dsigmaxz_dz
                      // memory_dsigmaxy_dx, memory_dsigmayy_dy, memory_dsigmayz_dz
                      // memory_dsigmaxz_dx, memory_dsigmayz_dy, memory_dsigmazz_dz
var<storage,read_write> memo: array<f32>;

// Group 2 - sensors arrays and energies
@group(2) @binding(9) // sensors signals (sisvx, sisvy, sisvz)
var<storage,read_write> sensors: array<f32>;

@group(2) @binding(10) // epsilon fields
                       // epsilon_xx, epsilon_yy, epsilon_zz, epsilon_xy, epsilon_xz, epsilon_yz, v_2
var<storage,read_write> eps: array<f32>;

@group(2) @binding(11) // energy fields
                       // total_energy, total_energy_kinetic, total_energy_potential, v_solid_norm
var<storage,read_write> energy: array<f32>;


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
// function to get a force_x array value [force_x]
fn get_force_x(n: i32) -> f32 {
    let index: i32 = ij(n, 0, sim_int_par.n_iter, 2);

    return select(0.0, force[index], index != -1);
}

// function to get a force_y array value [force_y]
fn get_force_y(n: i32) -> f32 {
    let index: i32 = ij(n, 1, sim_int_par.n_iter, 2);

    return select(0.0, force[index], index != -1);
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
    let index: i32 = ij(0, n, 6, sim_int_par.y_sz - 2);

    return select(0.0, coef_y[index], index != -1);
}

// function to get a a_y_h array value
fn get_a_y_h(n: i32) -> f32 {
    let index: i32 = ij(3, n, 6, sim_int_par.y_sz - 2);

    return select(0.0, coef_y[index], index != -1);
}

// function to get a b_y array value
fn get_b_y(n: i32) -> f32 {
    let index: i32 = ij(1, n, 6, sim_int_par.y_sz - 2);

    return select(0.0, coef_y[index], index != -1);
}

// function to get a b_y_h array value
fn get_b_y_h(n: i32) -> f32 {
    let index: i32 = ij(4, n, 6, sim_int_par.y_sz - 2);

    return select(0.0, coef_y[index], index != -1);
}

// function to get a k_y array value
fn get_k_y(n: i32) -> f32 {
    let index: i32 = ij(2, n, 6, sim_int_par.y_sz - 2);

    return select(0.0, coef_y[index], index != -1);
}

// function to get a k_y_h array value
fn get_k_y_h(n: i32) -> f32 {
    let index: i32 = ij(5, n, 6, sim_int_par.y_sz - 2);

    return select(0.0, coef_y[index], index != -1);
}

// -------------------------------------------------
// --- CPML Z coefficients array access funtions ---
// -------------------------------------------------
// function to get a a_z array value
fn get_a_z(n: i32) -> f32 {
    let index: i32 = ij(0, n, 6, sim_int_par.z_sz - 2);

    return select(0.0, coef_z[index], index != -1);
}

// function to get a a_z_h array value
fn get_a_z_h(n: i32) -> f32 {
    let index: i32 = ij(3, n, 6, sim_int_par.z_sz - 2);

    return select(0.0, coef_z[index], index != -1);
}

// function to get a b_z array value
fn get_b_z(n: i32) -> f32 {
    let index: i32 = ij(1, n, 6, sim_int_par.z_sz - 2);

    return select(0.0, coef_z[index], index != -1);
}

// function to get a b_z_h array value
fn get_b_z_h(n: i32) -> f32 {
    let index: i32 = ij(4, n, 6, sim_int_par.z_sz - 2);

    return select(0.0, coef_z[index], index != -1);
}

// function to get a k_z array value
fn get_k_z(n: i32) -> f32 {
    let index: i32 = ij(2, n, 6, sim_int_par.z_sz - 2);

    return select(0.0, coef_z[index], index != -1);
}

// function to get a k_z_h array value
fn get_k_z_h(n: i32) -> f32 {
    let index: i32 = ij(5, n, 6, sim_int_par.z_sz - 2);

    return select(0.0, coef_z[index], index != -1);
}

// ---------------------------------------
// --- Velocity arrays access funtions ---
// ---------------------------------------
// function to get a vx array value
fn get_vx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, vel[index], index != -1);
}

// function to set a vx array value
fn set_vx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        vel[index] = val;
    }
}

// function to get a vy array value
fn get_vy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, vel[index], index != -1);
}

// function to set a vy array value
fn set_vy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        vel[index] = val;
    }
}

// function to get a vz array value
fn get_vz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    return select(0.0, vel[index], index != -1);
}

// function to set a vz array value
fn set_vz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 3);

    if(index != -1) {
        vel[index] = val;
    }
}

// -------------------------------------
// --- Stress arrays access funtions ---
// -------------------------------------
// function to get a sigmaxx array value
fn get_sigmaxx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 6);

    return select(0.0, sig[index], index != -1);
}

// function to set a sigmaxx array value
fn set_sigmaxx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 6);

    if(index != -1) {
        sig[index] = val;
    }
}

// function to get a sigmayy array value
fn get_sigmayy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 6);

    return select(0.0, sig[index], index != -1);
}

// function to set a sigmayy array value
fn set_sigmayy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 6);

    if(index != -1) {
        sig[index] = val;
    }
}

// function to get a sigmazz array value
fn get_sigmazz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 6);

    return select(0.0, sig[index], index != -1);
}

// function to set a sigmazz array value
fn set_sigmazz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 6);

    if(index != -1) {
        sig[index] = val;
    }
}

// function to get a sigmaxy array value
fn get_sigmaxy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 3, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 6);

    return select(0.0, sig[index], index != -1);
}

// function to set a sigmaxy array value
fn set_sigmaxy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 3, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 6);

    if(index != -1) {
        sig[index] = val;
    }
}

// function to get a sigmaxz array value
fn get_sigmaxz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 4, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 6);

    return select(0.0, sig[index], index != -1);
}

// function to set a sigmaxz array value
fn set_sigmaxz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 4, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 6);

    if(index != -1) {
        sig[index] = val;
    }
}

// function to get a sigmayz array value
fn get_sigmayz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 5, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 6);

    return select(0.0, sig[index], index != -1);
}

// function to set a sigmayz array value
fn set_sigmayz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 5, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 6);

    if(index != -1) {
        sig[index] = val;
    }
}

// -------------------------------------
// --- Memory arrays access funtions ---
// -------------------------------------
// function to get a memory_dvx_dx array value
fn get_mdvx_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvx_dx array value
fn set_mdvx_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dvy_dx array value
fn get_mdvy_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvy_dx array value
fn set_mdvy_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dvz_dx array value
fn get_mdvz_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvz_dx array value
fn set_mdvz_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dvx_dy array value
fn get_mdvx_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 3, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvx_dy array value
fn set_mdvx_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 3, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dvy_dy array value
fn get_mdvy_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 4, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvy_dy array value
fn set_mdvy_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 4, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dvz_dy array value
fn get_mdvz_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 5, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvz_dy array value
fn set_mdvz_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 5, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dvx_dz array value
fn get_mdvx_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 6, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvy_dx array value
fn set_mdvx_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 6, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dvy_dz array value
fn get_mdvy_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 7, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvy_dz array value
fn set_mdvy_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 7, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dvz_dz array value
fn get_mdvz_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 8, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dvz_dz array value
fn set_mdvz_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 8, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmaxx_dx array value
fn get_mdsxx_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 9, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmaxx_dx array value
fn set_mdsxx_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 9, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmaxy_dy array value
fn get_mdsxy_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 10, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmaxy_dy array value
fn set_mdsxy_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 10, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmaxz_dz array value
fn get_mdsxz_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 11, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmaxz_dz array value
fn set_mdsxz_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 11, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmaxy_dx array value
fn get_mdsxy_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 12, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmaxy_dx array value
fn set_mdsxy_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 12, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmayy_dy array value
fn get_mdsyy_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 13, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmayy_dy array value
fn set_mdsyy_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 13, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmayz_dz array value
fn get_mdsyz_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 14, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmayz_dz array value
fn set_mdsyz_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 14, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmaxz_dx array value
fn get_mdsxz_dx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 15, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmaxz_dx array value
fn set_mdsxz_dx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 15, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmayz_dy array value
fn get_mdsyz_dy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 16, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmayz_dy array value
fn set_mdsyz_dy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 16, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// function to get a memory_dsigmazz_dz array value
fn get_mdszz_dz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 17, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    return select(0.0, memo[index], index != -1);
}

// function to set a memory_dsigmazz_dz array value
fn set_mdszz_dz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 17, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 18);

    if(index != -1) {
        memo[index] = val;
    }
}

// --------------------------------------
// --- Sensors arrays access funtions ---
// --------------------------------------
// function to set a sens_vx array value
fn set_sens_vx(n: i32, val: f32) {
    let index: i32 = ij(n, 0, sim_int_par.n_iter, 3);

    if(index != -1) {
        sensors[index] = val;
    }
}

// function to set a sens_vy array value
fn set_sens_vy(n: i32, val: f32) {
    let index: i32 = ij(n, 1, sim_int_par.n_iter, 3);

    if(index != -1) {
        sensors[index] = val;
    }
}

// function to set a sens_vz array value
fn set_sens_vz(n: i32, val: f32) {
    let index: i32 = ij(n, 1, sim_int_par.n_iter, 3);

    if(index != -1) {
        sensors[index] = val;
    }
}

// -------------------------------------
// --- Epsilon arrays access funtions ---
// -------------------------------------
// function to get a epsilon_xx array value
fn get_eps_xx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    return select(0.0, eps[index], index != -1);
}

// function to set a epsilon_xx array value
fn set_eps_xx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 0, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    if(index != -1) {
        eps[index] = val;
    }
}

// function to get a epsilon_yy array value
fn get_eps_yy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    return select(0.0, eps[index], index != -1);
}

// function to set a epsilon_yy array value
fn set_eps_yy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 1, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    if(index != -1) {
        eps[index] = val;
    }
}

// function to get a epsilon_zz array value
fn get_eps_zz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    return select(0.0, eps[index], index != -1);
}

// function to set a epsilon_zz array value
fn set_eps_zz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 2, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    if(index != -1) {
        eps[index] = val;
    }
}

// function to get a epsilon_xy array value
fn get_eps_xy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 3, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    return select(0.0, eps[index], index != -1);
}

// function to set a epsilon_xy array value
fn set_eps_xy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 3, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    if(index != -1) {
        eps[index] = val;
    }
}

// function to get a epsilon_xz array value
fn get_eps_xz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 4, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    return select(0.0, eps[index], index != -1);
}

// function to set a epsilon_xz array value
fn set_eps_xz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 4, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    if(index != -1) {
        eps[index] = val;
    }
}

// function to get a epsilon_yz array value
fn get_eps_yz(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 5, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    return select(0.0, eps[index], index != -1);
}

// function to set a epsilon_yz array value
fn set_eps_yz(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 5, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    if(index != -1) {
        eps[index] = val;
    }
}

// function to get a v_2 array value
fn get_v_2(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijkl(x, y, z, 6, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    return select(0.0, eps[index], index != -1);
}

// function to set a v_2 array value
fn set_v_2(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijkl(x, y, z, 6, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.z_sz, 7);

    if(index != -1) {
        eps[index] = val;
    }
}

// ------------------------------------
// --- Energy array access funtions ---
// ------------------------------------
// function to get a total_energy array value [tot_en]
fn get_tot_en(n: i32) -> f32 {
    let index: i32 = ij(n, 0, sim_int_par.n_iter, 4);

    return select(0.0, energy[index], index != -1);
}

// function to set a total_energy array value
fn set_tot_en(n: i32, val: f32) {
    let index: i32 = ij(n, 0, sim_int_par.n_iter, 4);

    if(index != -1) {
        energy[index] = val;
    }
}

// function to get a total_energy_kinetic array value [tot_en_k]
fn get_tot_en_k(n: i32) -> f32 {
    let index: i32 = ij(n, 1, sim_int_par.n_iter, 4);

    return select(0.0, energy[index], index != -1);
}

// function to set a total_energy_kinetic array value
fn set_tot_en_k(n: i32, val: f32) {
    let index: i32 = ij(n, 1, sim_int_par.n_iter, 4);

    if(index != -1) {
        energy[index] = val;
    }
}

// function to get a total_energy_potencial array value [tot_en_p]
fn get_tot_en_p(n: i32) -> f32 {
    let index: i32 = ij(n, 2, sim_int_par.n_iter, 4);

    return select(0.0, energy[index], index != -1);
}

// function to set a total_energy_potencial array value
fn set_tot_en_p(n: i32, val: f32) {
    let index: i32 = ij(n, 2, sim_int_par.n_iter, 4);

    if(index != -1) {
        energy[index] = val;
    }
}

// function to get a v_solid_norm array value [v_sol_n]
fn get_v_sol_n(n: i32) -> f32 {
    let index: i32 = ij(n, 3, sim_int_par.n_iter, 4);

    return select(0.0, energy[index], index != -1);
}

// function to set a v_solid_norm array value
fn set_v_sol_n(n: i32, val: f32) {
    let index: i32 = ij(n, 3, sim_int_par.n_iter, 4);

    if(index != -1) {
        energy[index] = val;
    }
}

// ---------------
// --- Kernels ---
// ---------------
// Kernel to calculate stresses [sigmaxx, sigmayy, sigmazz, sigmaxy, sigmaxz, sigmayz]
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

    // Normal stresses
    if(x >= 1 && x < (sim_int_par.x_sz - 2) &&
       y >= 2 && y < (sim_int_par.y_sz - 1) &&
       z >= 2 && z < (sim_int_par.z_sz - 1)) {
        //var vdvx_dx: f32 = (27.0*(get_vx(x + 1, y) - get_vx(x, y)) - get_vx(x + 2, y) + get_vx(x - 1, y))/(24.0 * dx);
        //var vdvy_dy: f32 = (27.0*(get_vy(x, y) - get_vy(x, y - 1)) - get_vy(x, y + 1) + get_vy(x, y - 2))/(24.0 * dy);

        //var mdvx_dx_new: f32 = get_b_x_h(x - 1) * get_mdvx_dx(x, y) + get_a_x_h(x - 1) * vdvx_dx;
        //var mdvy_dy_new: f32 = get_b_y(y - 1) * get_mdvy_dy(x, y) + get_a_y(y - 1) * vdvy_dy;

        //vdvx_dx = vdvx_dx/get_k_x_h(x - 1) + mdvx_dx_new;
        //vdvy_dy = vdvy_dy/get_k_y(y - 1)  + mdvy_dy_new;

        //set_mdvx_dx(x, y, mdvx_dx_new);
        //set_mdvy_dy(x, y, mdvy_dy_new);

        //set_sigmaxx(x, y, get_sigmaxx(x, y) + (lambdaplus2mu * vdvx_dx + lambda        * vdvy_dy) * dt);
        //set_sigmayy(x, y, get_sigmayy(x, y) + (lambda        * vdvx_dx + lambdaplus2mu * vdvy_dy) * dt);
    }

    // Shear stress
    if(x >= 2 && x < (sim_int_par.x_sz - 1) &&
       y >= 1 && y < (sim_int_par.y_sz - 2) &&
       z >= 2 && z < (sim_int_par.z_sz - 1)) {
        //var vdvy_dx: f32 = (27.0*(get_vy(x, y) - get_vy(x - 1, y)) - get_vy(x + 1, y) + get_vy(x - 2, y))/(24.0 * dx);
        //var vdvx_dy: f32 = (27.0*(get_vx(x, y + 1) - get_vx(x, y)) - get_vx(x, y + 2) + get_vx(x, y - 1))/(24.0 * dy);

        //var mdvy_dx_new: f32 = get_b_x(x - 1) * get_mdvy_dx(x, y) + get_a_x(x - 1) * vdvy_dx;
        //var mdvx_dy_new: f32 = get_b_y_h(y - 1) * get_mdvx_dy(x, y) + get_a_y_h(y - 1) * vdvx_dy;

        //vdvy_dx = vdvy_dx/get_k_x(x - 1)   + mdvy_dx_new;
        //vdvx_dy = vdvx_dy/get_k_y_h(y - 1) + mdvx_dy_new;

        //set_mdvy_dx(x, y, mdvy_dx_new);
        //set_mdvx_dy(x, y, mdvx_dy_new);

        //set_sigmaxy(x, y, get_sigmaxy(x, y) + (vdvx_dy + vdvy_dx) * mu * dt);
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

    // first step
    if(x >= 2 && x < (sim_int_par.x_sz - 1) &&
       y >= 2 && y < (sim_int_par.y_sz - 1) &&
       z >= 2 && z < (sim_int_par.z_sz - 1)) {
        //var vdsigmaxx_dx: f32 = (27.0*(get_sigmaxx(x, y) - get_sigmaxx(x - 1, y)) - get_sigmaxx(x + 1, y) + get_sigmaxx(x - 2, y))/(24.0 * dx);
        //var vdsigmaxy_dy: f32 = (27.0*(get_sigmaxy(x, y) - get_sigmaxy(x, y - 1)) - get_sigmaxy(x, y + 1) + get_sigmaxy(x, y - 2))/(24.0 * dy);

        //var mdsxx_dx_new: f32 = get_b_x(x - 1) * get_mdsxx_dx(x, y) + get_a_x(x - 1) * vdsigmaxx_dx;
        //var mdsxy_dy_new: f32 = get_b_y(y - 1) * get_mdsxy_dy(x, y) + get_a_y(y - 1) * vdsigmaxy_dy;

        //vdsigmaxx_dx = vdsigmaxx_dx/get_k_x(x - 1) + mdsxx_dx_new;
        //vdsigmaxy_dy = vdsigmaxy_dy/get_k_y(y - 1) + mdsxy_dy_new;

        //set_mdsxx_dx(x, y, mdsxx_dx_new);
        //set_mdsxy_dy(x, y, mdsxy_dy_new);

        //set_vx(x, y, dt_over_rho * (vdsigmaxx_dx + vdsigmaxy_dy) + get_vx(x, y));
    }

    // second step
    if(x >= 1 && x < (sim_int_par.x_sz - 2) &&
       y >= 1 && y < (sim_int_par.y_sz - 2) &&
       z >= 2 && z < (sim_int_par.z_sz - 1)) {
        //var vdsigmaxy_dx: f32 = (27.0*(get_sigmaxy(x + 1, y) - get_sigmaxy(x, y)) - get_sigmaxy(x + 2, y) + get_sigmaxy(x - 1, y))/(24.0 * dx);
        //var vdsigmayy_dy: f32 = (27.0*(get_sigmayy(x, y + 1) - get_sigmayy(x, y)) - get_sigmayy(x, y + 2) + get_sigmayy(x, y - 1))/(24.0 * dy);

        //var mdsxy_dx_new: f32 = get_b_x_h(x - 1) * get_mdsxy_dx(x, y) + get_a_x_h(x - 1) * vdsigmaxy_dx;
        //var mdsyy_dy_new: f32 = get_b_y_h(y - 1) * get_mdsyy_dy(x, y) + get_a_y_h(y - 1) * vdsigmayy_dy;

        //vdsigmaxy_dx = vdsigmaxy_dx/get_k_x_h(x - 1) + mdsxy_dx_new;
        //vdsigmayy_dy = vdsigmayy_dy/get_k_y_h(y - 1) + mdsyy_dy_new;

        //set_mdsxy_dx(x, y, mdsxy_dx_new);
        //set_mdsyy_dy(x, y, mdsyy_dy_new);

        //set_vy(x, y, dt_over_rho * (vdsigmaxy_dx + vdsigmayy_dy) + get_vy(x, y));
    }
}

// Kernel to finish iteration term
@compute
@workgroup_size(wsx, wsy, wsz)
fn finish_it_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let z: i32 = i32(index.z);          // y thread index
    let dt_over_rho: f32 = sim_flt_par.dt / sim_flt_par.rho;
    let x_source: i32 = sim_int_par.x_source;
    let y_source: i32 = sim_int_par.y_source;
    let z_source: i32 = sim_int_par.z_source;
    let x_sens: i32 = sim_int_par.x_sens;
    let y_sens: i32 = sim_int_par.y_sens;
    let z_sens: i32 = sim_int_par.z_sens;
    let it: i32 = sim_int_par.it;
    let xmin: i32 = sim_int_par.np_pml;
    let xmax: i32 = sim_int_par.x_sz - sim_int_par.np_pml;
    let ymin: i32 = sim_int_par.np_pml;
    let ymax: i32 = sim_int_par.y_sz - sim_int_par.np_pml;
    let zmin: i32 = sim_int_par.np_pml;
    let zmax: i32 = sim_int_par.z_sz - sim_int_par.np_pml;
    let rho_2: f32 = sim_flt_par.rho * 0.5;
    let lambda: f32 = sim_flt_par.lambda;
    let mu: f32 = sim_flt_par.mu;
    let lambdaplus2mu: f32 = lambda + 2.0 * mu;
    let denom: f32 = 4.0 * mu * (lambda + mu);
    let mu2: f32 = 2.0 * mu;

    // Add the source force
    if(x == x_source && y == y_source && z == z_source) {
        //set_vx(x, y, get_vx(x, y) + get_force_x(it) * dt_over_rho);
        //set_vy(x, y, get_vy(x, y) + get_force_y(it) * dt_over_rho);
    }

    // Apply Dirichlet conditions
    if(x <= 1 || x >= (sim_int_par.x_sz - 2) ||
       y <= 1 || y >= (sim_int_par.y_sz - 2) ||
       z <= 1 || z >= (sim_int_par.z_sz - 2)) {
        //set_vx(x, y, 0.0);
        //set_vy(x, y, 0.0);
    }

    // Store sensor velocities
    if(x == x_sens && y == y_sens && z == z_sens) {
        //set_sens_vx(it, get_vx(x, y));
        //set_sens_vy(it, get_vy(x, y));
    }

    // Compute total energy in the medium (without the PML layers)
    //set_v_2(x, y, get_vx(x, y)*get_vx(x, y) + get_vy(x, y)*get_vy(x, y));
    if(x >= xmin && x < xmax &&
       y >= ymin && y < ymax &&
       z >= zmin && z < zmax) {
        //set_tot_en_k(it, rho_2 * get_vx_vy_2(x, y) + get_tot_en_k(it));

        //set_eps_xx(x, y, (lambdaplus2mu*get_sigmaxx(x, y) - lambda*get_sigmayy(x, y))/denom);
        //set_eps_yy(x, y, (lambdaplus2mu*get_sigmayy(x, y) - lambda*get_sigmaxx(x, y))/denom);
        //set_eps_xy(x, y, get_sigmaxy(x, y)/mu2);

        //set_tot_en_p(it, 0.5 * (get_eps_xx(x, y)*get_sigmaxx(x, y) +
        //                        get_eps_yy(x, y)*get_sigmayy(x, y) +
        //                        2.0 * get_eps_xy(x, y)* get_sigmaxy(x, y)));

        //set_tot_en(it, get_tot_en_k(it) + get_tot_en_p(it));
    }
}

// Kernel to increase time iteraction [it]
@compute
@workgroup_size(1)
fn incr_it_kernel() {
    sim_int_par.it += 1;
}
