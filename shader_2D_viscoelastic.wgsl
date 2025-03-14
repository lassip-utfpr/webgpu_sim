// +++++++++++++++++++++++++++++++
// +++ Index access functions ++++
// +++++++++++++++++++++++++++++++
// function to convert 2D [i,j] index into 1D [] index
fn ij(i: i32, j: i32, i_max: i32, j_max: i32) -> i32 {
    let index = j + i * j_max;

    return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max);
}

// function to convert 3D [i,j,k] index into 1D [] index
fn ijk(i: i32, j: i32, k: i32, i_max: i32, j_max: i32, k_max: i32) -> i32 {
    let index = k + j * k_max + i * k_max * j_max;

    return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max);
}
// ++++++++++++++++++++++++++++++
// ++++ Group 0 - parameters ++++
// ++++++++++++++++++++++++++++++
struct SimIntValues {
    x_sz: i32,          // x field size
    y_sz: i32,          // y field size
    n_iter: i32,        // num iterations
    n_src_el: i32,      // num probes tx elements
    n_rec_el: i32,      // num probes rx elements
    n_rec_pt: i32,      // num rec pto
    fd_coeff: i32,      // num fd coefficients
    it: i32,            // time iteraction
    n_sls: i32,         // num zener bodies
    visco_attn: i32     // viscoelastic attenuation flag
};

@group(0) @binding(0) // param_int32
var<storage,read_write> sim_int_par: SimIntValues;

// ----------------------------------

struct SimFltValues {
    dx: f32,            // delta x
    dy: f32,            // delta y
    dt: f32,            // delta t
};

@group(0) @binding(1)   // param_flt32
var<storage,read> sim_flt_par: SimFltValues;

// -----------------------------------
// --- Force array access funtions ---
// -----------------------------------
@group(0) @binding(2) // source term
var<storage,read> source_term: array<f32>;

// function to get a source_term array value
fn get_source_term(n: i32, e: i32) -> f32 {
    let index: i32 = ij(n, e, sim_int_par.n_iter, sim_int_par.n_src_el);

    return select(0.0, source_term[index], index != -1);
}

// ----------------------------------

@group(0) @binding(3) // source term index
var<storage,read> idx_src: array<i32>;

// function to get a source term index of a source
fn get_idx_source_term(x: i32, y: i32) -> i32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(-1, idx_src[index], index != -1);
}

// -------------------------------------------------
// --- CPML X coefficients array access funtions ---
// -------------------------------------------------
@group(0) @binding(4) // a_x
var<storage,read> a_x: array<f32>;

// function to get a a_x array value
fn get_a_x(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_x[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(5) // b_x
var<storage,read> b_x: array<f32>;

// function to get a b_x array value
fn get_b_x(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_x[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(6) // k_x
var<storage,read> k_x: array<f32>;

// function to get a k_x array value
fn get_k_x(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_x[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(7) // a_x_h
var<storage,read> a_x_h: array<f32>;

// function to get a a_x_h array value
fn get_a_x_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_x_h[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(8) // b_x_h
var<storage,read> b_x_h: array<f32>;

// function to get a b_x_h array value
fn get_b_x_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_x_h[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// ----------------------------------

@group(0) @binding(9) // k_x_h
var<storage,read> k_x_h: array<f32>;

// function to get a k_x_h array value
fn get_k_x_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_x_h[n], n >= 0 && n < sim_int_par.x_sz - pad);
}

// -------------------------------------------------
// --- CPML Y coefficients array access funtions ---
// -------------------------------------------------
@group(0) @binding(10) // a_y
var<storage,read> a_y: array<f32>;

// function to get a a_y array value
fn get_a_y(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_y[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(11) // b_y
var<storage,read> b_y: array<f32>;

// function to get a b_y array value
fn get_b_y(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_y[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(12) // k_y
var<storage,read> k_y: array<f32>;

// function to get a k_y array value
fn get_k_y(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_y[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(13) // a_y_h
var<storage,read> a_y_h: array<f32>;

// function to get a a_y_h array value
fn get_a_y_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, a_y_h[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(14) // b_y_h
var<storage,read> b_y_h: array<f32>;

// function to get a b_y_h array value
fn get_b_y_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, b_y_h[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// ----------------------------------

@group(0) @binding(15) // k_y_h
var<storage,read> k_y_h: array<f32>;

// function to get a k_y_h array value
fn get_k_y_h(n: i32) -> f32 {
    let pad: i32 = (sim_int_par.fd_coeff - 1) * 2;

    return select(0.0, k_y_h[n], n >= 0 && n < sim_int_par.y_sz - pad);
}

// -------------------------------------------------------------
// --- Finite difference index limits arrays access funtions ---
// -------------------------------------------------------------
@group(0) @binding(16) // idx_fd
var<storage,read> idx_fd: array<i32>;

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

// ----------------------------------

@group(0) @binding(17) // fd_coeff
var<storage,read> fd_coeffs: array<f32>;

// function to get a fd coefficient
fn get_fdc(c: i32) -> f32 {
    return select(0.0, fd_coeffs[c], c >= 0 && c < sim_int_par.fd_coeff);
}

// ---------------------------------
// --- Rho map access funtions ---
// ---------------------------------
@group(0) @binding(18) // rho
var<storage,read> rho_map: array<f32>;

// function to get a rho value
fn get_rho(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, rho_map[index], index != -1);
}

// ---------------------------------
// --- Cp map access funtions ---
// ---------------------------------
@group(0) @binding(19) // cp
var<storage,read> cp_map: array<f32>;

// function to get a cp value
fn get_cp(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, cp_map[index], index != -1);
}

// ---------------------------------
// --- Cs map access funtions ---
// ---------------------------------
@group(0) @binding(20) // cs
var<storage,read> cs_map: array<f32>;

// function to get a cp value
fn get_cs(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, cs_map[index], index != -1);
}

@group(0) @binding(21) //r_xx
var<storage,read_write> r_xx: array<f32>;

fn get_r_xx(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.n_sls);

    return select(0.0, r_xx[index], index != -1);
}

// function to set a vx array value
fn set_r_xx(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_xx[index] = val;
    }
}
@group(0) @binding(22) //somatorio alpha_p e alpha_s
var<storage,read> sum_alpha: array<f32>;

@group(0) @binding(23) //tau_epsilon_nu1,tau_sigma_nu1,tau_epsilon_nu2, tau_sigma_nu2 / o x indica qual tau e o y indica qual corpo zener
var<storage,read> tau_att: array<f32>;

fn get_tau(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, 4, sim_int_par.n_sls);

    return select(0.0, tau_att[index], index != -1);
}

@group(0) @binding(24) //r_yy
var<storage,read_write> r_yy: array<f32>;

fn get_r_yy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.n_sls);

    return select(0.0, r_yy[index], index != -1);
}

// function to set a vx array value
fn set_r_yy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_yy[index] = val;
    }
}

@group(0) @binding(25) //r_xy
var<storage,read_write> r_xy: array<f32>;

fn get_r_xy(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.n_sls);

    return select(0.0, r_xy[index], index != -1);
}

// function to set a vx array value
fn set_r_xy(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_xy[index] = val;
    }
}

@group(0) @binding(26) //r_xx_old
var<storage,read_write> r_xx_old: array<f32>;

fn get_r_xx_old(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.n_sls);

    return select(0.0, r_xx_old[index], index != -1);
}

// function to set a vx array value
fn set_r_xx_old(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_xx_old[index] = val;
    }
}

@group(0) @binding(27) //r_yy_old
var<storage,read_write> r_yy_old: array<f32>;

fn get_r_yy_old(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.n_sls);

    return select(0.0, r_yy_old[index], index != -1);
}

// function to set a vx array value
fn set_r_yy_old(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_yy_old[index] = val;
    }
}

@group(0) @binding(28) //r_xy_old
var<storage,read_write> r_xy_old: array<f32>;

fn get_r_xy_old(x: i32, y: i32, z: i32) -> f32 {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.n_sls);

    return select(0.0, r_xy_old[index], index != -1);
}

// function to set a vx array value
fn set_r_xy_old(x: i32, y: i32, z: i32, val: f32) {
    let index: i32 = ijk(x, y, z, sim_int_par.x_sz, sim_int_par.y_sz, sim_int_par.n_sls);

    if(index != -1) {
        r_xy_old[index] = val;
    }
}

// +++++++++++++++++++++++++++++++++++++
// ++++ Group 1 - simulation arrays ++++
// +++++++++++++++++++++++++++++++++++++
// ---------------------------------------
// --- Velocity arrays access funtions ---
// ---------------------------------------
@group(1) @binding(0) // vx field
var<storage,read_write> vx: array<f32>;

// function to get a vx array value
fn get_vx(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, vx[index], index != -1);
}

// function to set a vx array value
fn set_vx(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        vx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(1) // vy field
var<storage,read_write> vy: array<f32>;

// function to get a vy array value
fn get_vy(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, vy[index], index != -1);
}

// function to set a vy array value
fn set_vy(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        vy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(2) // v_2
var<storage,read_write> v_2: f32;

// -------------------------------------
// --- Stress arrays access funtions ---
// -------------------------------------
@group(1) @binding(3) // sigmaxx field
var<storage,read_write> sigmaxx: array<f32>;

// function to get a sigmaxx array value
fn get_sigmaxx(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, sigmaxx[index], index != -1);
}

// function to set a sigmaxx array value
fn set_sigmaxx(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        sigmaxx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(4) // sigmayy field
var<storage,read_write> sigmayy: array<f32>;

// function to get a sigmayy array value
fn get_sigmayy(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, sigmayy[index], index != -1);
}

// function to set a sigmayy array value
fn set_sigmayy(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        sigmayy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(5) // sigmaxy field
var<storage,read_write> sigmaxy: array<f32>;

// function to get a sigmaxy array value
fn get_sigmaxy(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, sigmaxy[index], index != -1);
}

// function to set a sigmaxy array value
fn set_sigmaxy(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        sigmaxy[index] = val;
    }
}

// -------------------------------------
// --- Memory arrays access funtions ---
// -------------------------------------
@group(1) @binding(6) // mdvx_dx field
var<storage,read_write> mdvx_dx: array<f32>;

// function to get a memory_dvx_dx array value
fn get_mdvx_dx(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, mdvx_dx[index], index != -1);
}

// function to set a memory_dvx_dx array value
fn set_mdvx_dx(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        mdvx_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(7) // mdvx_dy field
var<storage,read_write> mdvx_dy: array<f32>;

// function to get a memory_dvx_dy array value
fn get_mdvx_dy(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, mdvx_dy[index], index != -1);
}

// function to set a memory_dvx_dy array value
fn set_mdvx_dy(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        mdvx_dy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(8) // mdvy_dx field
var<storage,read_write> mdvy_dx: array<f32>;

// function to get a memory_dvy_dx array value
fn get_mdvy_dx(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, mdvy_dx[index], index != -1);
}

// function to set a memory_dvy_dx array value
fn set_mdvy_dx(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        mdvy_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(9) // mdvy_dy field
var<storage,read_write> mdvy_dy: array<f32>;

// function to get a memory_dvy_dy array value
fn get_mdvy_dy(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, mdvy_dy[index], index != -1);
}

// function to set a memory_dvy_dy array value
fn set_mdvy_dy(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        mdvy_dy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(10) // mdsxx_dx field
var<storage,read_write> mdsxx_dx: array<f32>;

// function to get a memory_dsigmaxx_dx array value
fn get_mdsxx_dx(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, mdsxx_dx[index], index != -1);
}

// function to set a memory_dsigmaxx_dx array value
fn set_mdsxx_dx(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        mdsxx_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(11) // mdsyy_dy field
var<storage,read_write> mdsyy_dy: array<f32>;

// function to get a memory_dsigmayy_dy array value
fn get_mdsyy_dy(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, mdsyy_dy[index], index != -1);
}

// function to set a memory_dsigmayy_dy array value
fn set_mdsyy_dy(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        mdsyy_dy[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(12) // mdsxy_dx field
var<storage,read_write> mdsxy_dx: array<f32>;

// function to get a memory_dsigmaxy_dx array value
fn get_mdsxy_dx(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, mdsxy_dx[index], index != -1);
}

// function to set a memory_dsigmaxy_dx array value
fn set_mdsxy_dx(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        mdsxy_dx[index] = val;
    }
}

// ----------------------------------

@group(1) @binding(13) // mdsxy_dy field
var<storage,read_write> mdsxy_dy: array<f32>;

// function to get a memory_dsigmaxy_dy array value
fn get_mdsxy_dy(x: i32, y: i32) -> f32 {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    return select(0.0, mdsxy_dy[index], index != -1);
}

// function to set a memory_dsigmaxy_dy array value
fn set_mdsxy_dy(x: i32, y: i32, val : f32) {
    let index: i32 = ij(x, y, sim_int_par.x_sz, sim_int_par.y_sz);

    if(index != -1) {
        mdsxy_dy[index] = val;
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++
// ++++ Group 2 - sensors arrays and energies ++++
// +++++++++++++++++++++++++++++++++++++++++++++++
// --------------------------------------
// --- Sensors arrays access funtions ---
// --------------------------------------
@group(2) @binding(0) // sensors signals vx
var<storage,read_write> sensors_vx: array<f32>;

// function to get a sens_vx array value
fn get_sens_vx(n: i32, s: i32) -> f32 {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    return select(0.0, sensors_vx[index], index != -1);
}

// function to set a sens_vx array value
fn set_sens_vx(n: i32, s: i32, val : f32) {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    if(index != -1) {
        sensors_vx[index] = val;
    }
}

// ----------------------------------

@group(2) @binding(1) // sensors signals vy
var<storage,read_write> sensors_vy: array<f32>;

// function to get a sens_vy array value
fn get_sens_vy(n: i32, s: i32) -> f32 {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    return select(0.0, sensors_vy[index], index != -1);
}

// function to set a sens_vy array value
fn set_sens_vy(n: i32, s: i32, val : f32) {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    if(index != -1) {
        sensors_vy[index] = val;
    }
}

// ----------------------------------

@group(2) @binding(2) // delay sensor
var<storage,read> delay_rec: array<i32>;

// function to get a delay receiver value
fn get_delay_rec(s: i32) -> i32 {
    return select(0, delay_rec[s], s >= 0 && s < sim_int_par.n_rec_el);
}

// ----------------------------------

@group(2) @binding(3) // info rec ptos
var<storage,read> info_rec_pt: array<i32>;

// function to get a x-index of a receiver point
fn get_idx_x_sensor(n: i32) -> i32 {
    let index: i32 = ij(n, 0, sim_int_par.n_rec_pt, 3);

    return select(-1, info_rec_pt[index], index != -1);
}

// function to get a y-index of a receiver point
fn get_idx_y_sensor(n: i32) -> i32 {
    let index: i32 = ij(n, 1, sim_int_par.n_rec_pt, 3);

    return select(-1, info_rec_pt[index], index != -1);
}

// function to get a sensor-index of a receiver point
fn get_idx_sensor(n: i32) -> i32 {
    let index: i32 = ij(n, 2, sim_int_par.n_rec_pt, 3);

    return select(-1, info_rec_pt[index], index != -1);
}

// ----------------------------------

@group(2) @binding(4) // info rec ptos
var<storage,read> offset_sensors: array<i32>;

// function to get the offset of a sensor receiver in info_rec_pt table
fn get_offset_sensor(s: i32) -> i32 {
    return select(-1, offset_sensors[s], s >= 0 && s < sim_int_par.n_rec_el);
}

// ----------------------------------

@group(2) @binding(5) // sensors signals sigxx
var<storage,read_write> sensors_sigxx: array<f32>;

// function to get a sens_sigxx array value
fn get_sens_sigxx(n: i32, s: i32) -> f32 {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    return select(0.0, sensors_sigxx[index], index != -1);
}

// function to set a sens_sigxx array value
fn set_sens_sigxx(n: i32, s: i32, val : f32) {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    if(index != -1) {
        sensors_sigxx[index] = val;
    }
}

// ----------------------------------

@group(2) @binding(6) // sensors signals sigyy
var<storage,read_write> sensors_sigyy: array<f32>;

// function to get a sens_sigyy array value
fn get_sens_sigyy(n: i32, s: i32) -> f32 {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    return select(0.0, sensors_sigyy[index], index != -1);
}

// function to set a sens_sigyy array value
fn set_sens_sigyy(n: i32, s: i32, val : f32) {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    if(index != -1) {
        sensors_sigyy[index] = val;
    }
}

// ----------------------------------

@group(2) @binding(7) // sensors signals sigxy
var<storage,read_write> sensors_sigxy: array<f32>;

// function to get a sens_sigxx array value
fn get_sens_sigxy(n: i32, s: i32) -> f32 {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    return select(0.0, sensors_sigxy[index], index != -1);
}

// function to set a sens_sigxy array value
fn set_sens_sigxy(n: i32, s: i32, val : f32) {
    let index: i32 = ij(n, s, sim_int_par.n_iter, sim_int_par.n_rec_el);

    if(index != -1) {
        sensors_sigxy[index] = val;
    }
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
    let last: i32 = sim_int_par.fd_coeff - 1;
    let offset_x: i32 = sim_int_par.fd_coeff - 1;
    let offset_y: i32 = sim_int_par.fd_coeff - 1;


    // Normal stresses
    var id_x_i: i32 = -get_idx_fh(last);
    var id_x_f: i32 = sim_int_par.x_sz - get_idx_ih(last);
    var id_y_i: i32 = -get_idx_ff(last);
    var id_y_f: i32 = sim_int_par.y_sz - get_idx_if(last);
    if(x >= id_x_i && x < id_x_f && y >= id_y_i && y < id_y_f) {
        set_vx(x, y, get_rho(x, y));
        set_vy(x, y, get_cp(x, y));
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
    let last: i32 = sim_int_par.fd_coeff - 1;
    let offset: i32 = sim_int_par.fd_coeff - 1;
    let visco_attn: bool = bool(sim_int_par.visco_attn);

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

        let rho_h_x = 0.5 * (get_rho(x + 1, y) + get_rho(x, y));
        let cp_h_x = 0.5 * (get_cp(x + 1, y) + get_cp(x, y));
        let cs_h_x_l = 0.5 * (get_cs(x + 1, y) + get_cs(x, y));
        let cs_h_x_m = select(cs_h_x_l, 0.0, min(get_cs(x + 1, y), get_cs(x, y)) == 0.0);
        let lambda: f32 = rho_h_x * (cp_h_x * cp_h_x - 2.0 * cs_h_x_l * cs_h_x_l);
        let mu: f32 = rho_h_x * (cs_h_x_m * cs_h_x_m);
        let lambdaplus2mu: f32 = lambda + 2.0 * mu;
        let lambdaplusmu: f32 = lambda + mu;
        var sigmaxx: f32 = 0.0;
        var sigmayy: f32 = 0.0;

        if(visco_attn) {

            var sum_r_xx: f32 = 0.0;
            var sum_r_yy: f32 = 0.0;

            for(var _l: i32 = 0; _l < sim_int_par.n_sls; _l++) {

                var inv_tau_sigma_p: f32 = 1.0/get_tau(1,_l);
                var inv_tau_sigma_s: f32 = 1.0/get_tau(3,_l);

                var alpha_p:f32 = get_tau(0,_l)/get_tau(1,_l);
                var alpha_s:f32 = get_tau(2,_l)/get_tau(3,_l);

                var deltat_phi_p: f32 = dt * (1.0 - alpha_p) * inv_tau_sigma_p / sum_alpha[0];
                var deltat_phi_s: f32 = dt * (1.0 - alpha_s) * inv_tau_sigma_s / sum_alpha[1];

                var half_deltat_overtau_sigma_p: f32 = 0.5 * dt * inv_tau_sigma_p;
                var half_deltat_overtau_sigma_s: f32 = 0.5 * dt * inv_tau_sigma_s;

                var mult_factor_tau_sigma_p: f32 = 1.0 / (1.0 + half_deltat_overtau_sigma_p);
                var mult_factor_tau_sigma_s: f32 = 1.0 / (1.0 + half_deltat_overtau_sigma_s);

                var r_xx_old: f32 = get_r_xx(x,y,_l);
                var r_yy_old: f32 = get_r_yy(x,y,_l);

                var r_xx: f32 = ((r_xx_old + (vdvx_dx + vdvy_dy) * deltat_phi_p - r_xx_old * half_deltat_overtau_sigma_p) * mult_factor_tau_sigma_p);
                var r_yy: f32 = ((r_yy_old + 0.5 * (vdvx_dx - vdvy_dy) * deltat_phi_s - r_yy_old * half_deltat_overtau_sigma_s) * mult_factor_tau_sigma_s);

                sum_r_xx += r_xx_old + r_xx;
                sum_r_yy += r_yy_old + r_yy;

                set_r_xx_old(x,y,_l,r_xx_old);
                set_r_xx(x,y,_l,r_xx);
                set_r_yy_old(x,y,_l,r_yy_old);
                set_r_yy(x,y,_l,r_yy);
            }

            sigmaxx = get_sigmaxx(x, y) + (lambdaplus2mu * vdvx_dx + lambda * vdvy_dy + 0.5 * lambdaplusmu * sum_r_xx + mu * sum_r_yy) * dt;
            sigmayy = get_sigmayy(x, y) + (lambda * vdvx_dx + lambdaplus2mu * vdvy_dy + 0.5 * lambdaplusmu * sum_r_xx - mu * sum_r_yy) * dt;
        }
        else {
            sigmaxx = get_sigmaxx(x, y) + (lambdaplus2mu * vdvx_dx + lambda        * vdvy_dy)*dt;
            sigmayy = get_sigmayy(x, y) + (lambda        * vdvx_dx + lambdaplus2mu * vdvy_dy)*dt;
        }
        set_sigmaxx(x, y, sigmaxx);
        set_sigmayy(x, y, sigmayy);
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

        let mdvy_dx_new: f32 = get_b_x(x - offset) * get_mdvy_dx(x, y) + get_a_x(x - offset) * vdvy_dx;
        let mdvx_dy_new: f32 = get_b_y_h(y - offset) * get_mdvx_dy(x, y) + get_a_y_h(y - offset) * vdvx_dy;

        vdvy_dx = vdvy_dx/get_k_x(x - offset)   + mdvy_dx_new;
        vdvx_dy = vdvx_dy/get_k_y_h(y - offset) + mdvx_dy_new;

        set_mdvy_dx(x, y, mdvy_dx_new);
        set_mdvx_dy(x, y, mdvx_dy_new);

        let rho_h_y = 0.5 * (get_rho(x, y + 1) + get_rho(x, y));
        let cs_h_y = select(0.5 * (get_cs(x, y + 1) + get_cs(x, y)), 0.0, min(get_cs(x, y + 1), get_cs(x, y)) == 0.0);
        let mu: f32 = rho_h_y * (cs_h_y * cs_h_y);
        var sigmaxy: f32 = 0.0;

        if(visco_attn) {

            var sum_r_xy: f32 = 0.0;

            for(var _l: i32 = 0; _l < sim_int_par.n_sls; _l++) {
                var inv_tau_sigma_s: f32 = 1.0/get_tau(3,_l);
                var alpha_s:f32 =  get_tau(2,_l) / get_tau(3,_l);
                var deltat_phi_s: f32 = dt * (1.0 - alpha_s) * inv_tau_sigma_s / sum_alpha[1];

                var half_deltat_overtau_sigma_s: f32 = 0.5 * dt * inv_tau_sigma_s;
                var mult_factor_tau_sigma_s: f32 = 1.0 / (1.0 + half_deltat_overtau_sigma_s);

                var r_xy_old: f32 = get_r_xy(x,y,_l);
                var r_xy: f32 = ((r_xy_old + (vdvy_dx + vdvx_dy) * deltat_phi_s - r_xy_old * half_deltat_overtau_sigma_s) * mult_factor_tau_sigma_s);

                sum_r_xy += r_xy_old + r_xy;

                set_r_xy_old(x,y,_l,r_xy_old);
                set_r_xy(x,y,_l,r_xy);
        }

            sigmaxy = get_sigmaxy(x, y) + (mu * (vdvy_dx + vdvx_dy) + 0.5 * sum_r_xy) * dt;
        }
        else {
            sigmaxy = get_sigmaxy(x, y) + (vdvx_dy + vdvy_dx) * mu * dt;
        }
        set_sigmaxy(x, y, sigmaxy);

    }
}

// Kernel to calculate velocities [vx, vy]
@compute
@workgroup_size(wsx, wsy)
fn velocity_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let dt: f32 = sim_flt_par.dt;
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

        let mdsxx_dx_new: f32 = get_b_x(x - offset) * get_mdsxx_dx(x, y) + get_a_x(x - offset) * vdsigmaxx_dx;
        let mdsxy_dy_new: f32 = get_b_y(y - offset) * get_mdsxy_dy(x, y) + get_a_y(y - offset) * vdsigmaxy_dy;

        vdsigmaxx_dx = vdsigmaxx_dx/get_k_x(x - offset) + mdsxx_dx_new;
        vdsigmaxy_dy = vdsigmaxy_dy/get_k_y(y - offset) + mdsxy_dy_new;

        set_mdsxx_dx(x, y, mdsxx_dx_new);
        set_mdsxy_dy(x, y, mdsxy_dy_new);

        let rho: f32 = get_rho(x, y);
        if(rho > 0.0) {
            let vx: f32 = (vdsigmaxx_dx + vdsigmaxy_dy) * dt / rho + get_vx(x, y);
            set_vx(x, y, vx);
        }
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

        let mdsxy_dx_new: f32 = get_b_x_h(x - offset) * get_mdsxy_dx(x, y) + get_a_x_h(x - offset) * vdsigmaxy_dx;
        let mdsyy_dy_new: f32 = get_b_y_h(y - offset) * get_mdsyy_dy(x, y) + get_a_y_h(y - offset) * vdsigmayy_dy;

        vdsigmaxy_dx = vdsigmaxy_dx/get_k_x_h(x - offset) + mdsxy_dx_new;
        vdsigmayy_dy = vdsigmayy_dy/get_k_y_h(y - offset) + mdsyy_dy_new;

        set_mdsxy_dx(x, y, mdsxy_dx_new);
        set_mdsyy_dy(x, y, mdsyy_dy_new);

        let rho: f32 = 0.25 * (get_rho(x, y) + get_rho(x + 1, y) + get_rho(x + 1, y + 1) + get_rho(x, y + 1));
        if(rho > 0.0) {
            let vy: f32 = (vdsigmaxy_dx + vdsigmayy_dy) * dt / rho + get_vy(x, y);
            set_vy(x, y, vy);
        }
    }
}

// Kernel to add the sources forces
@compute
@workgroup_size(wsx, wsy)
fn sources_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let dt: f32 = sim_flt_par.dt;
    let it: i32 = sim_int_par.it;

    // Add the source force
    let idx_src_term: i32 = get_idx_source_term(x, y);
    let rho: f32 = 0.25 * (get_rho(x, y) + get_rho(x + 1, y) + get_rho(x + 1, y + 1) + get_rho(x, y + 1));
    if(idx_src_term != -1 && rho > 0.0) {
        let val_src: f32 = get_source_term(it, idx_src_term);
        let vy: f32 = get_vy(x, y) +  val_src * dt / rho;
        set_vy(x, y, vy);
    }
}

// Kernel to finish iteration term
@compute
@workgroup_size(wsx, wsy)
fn finish_it_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let last: i32 = sim_int_par.fd_coeff - 1;
    let id_x_i: i32 = -get_idx_fh(last);
    let id_x_f: i32 = sim_int_par.x_sz - get_idx_ih(last);
    let id_y_i: i32 = -get_idx_fh(last);
    let id_y_f: i32 = sim_int_par.y_sz - get_idx_ih(last);
    let v_2_old: f32 = v_2;

    // Apply Dirichlet conditions
    if(x <= id_x_i || x >= id_x_f || y <= id_y_i || y >= id_y_f) {
        set_vx(x, y, 0.0);
        set_vy(x, y, 0.0);
    }

    // Compute velocity norm L2
    let v2: f32 = get_vx(x, y) * get_vx(x, y) + get_vy(x, y) * get_vy(x, y);
    v_2 = max(v_2_old, v2);
}

// Kernel to store sensors velocity
@compute
@workgroup_size(idx_rec_offset)
fn store_sensors_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
    let sensor: i32 = i32(index.x);          // x thread index
    let it: i32 = sim_int_par.it;

    // Store sensors velocities
    for(var pt: i32 = get_offset_sensor(sensor); get_idx_sensor(pt) == sensor; pt++) {
        if(it >= get_delay_rec(sensor)) {
            let x: i32 = get_idx_x_sensor(pt);
            let y: i32 = get_idx_y_sensor(pt);

            let value_sens_vx: f32 = get_sens_vx(it, sensor) + get_vx(x, y);
            let value_sens_vy: f32 = get_sens_vy(it, sensor) + get_vy(x, y);
            set_sens_vx(it, sensor, value_sens_vx);
            set_sens_vy(it, sensor, value_sens_vy);

            let value_sens_sigxx: f32 = get_sens_sigxx(it, sensor) + get_sigmaxx(x, y);
            let value_sens_sigyy: f32 = get_sens_sigyy(it, sensor) + get_sigmayy(x, y);
            let value_sens_sigxy: f32 = get_sens_sigxy(it, sensor) + get_sigmaxy(x, y);
            set_sens_sigxx(it, sensor, value_sens_sigxx);
            set_sens_sigyy(it, sensor, value_sens_sigyy);
            set_sens_sigxy(it, sensor, value_sens_sigxy);
        }
    }
}

// Kernel to increase time iteraction [it]
@compute
@workgroup_size(1)
fn incr_it_kernel() {
    sim_int_par.it += 1;
}
