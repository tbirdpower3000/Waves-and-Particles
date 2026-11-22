//! 2D FDTD with the same arithmetic as `src/fdtd.js`.
//! Single-threaded: GitHub Pages does not send COOP/COEP, so wasm threads
//! and SharedArrayBuffer are unavailable. Row-strip workers would copy the
//! halo every step and lose the gain on this stencil.

use wasm_bindgen::prelude::*;

const C0: f64 = 299792458.0;
const MU0: f64 = 4e-7 * std::f64::consts::PI;
const EPS0: f64 = 1.0 / (MU0 * C0 * C0);
const PML_M: i32 = 3;

#[wasm_bindgen]
pub fn c0() -> f64 {
    C0
}

#[wasm_bindgen]
pub fn courant_dt(dx: f64, dy: f64, c: Option<f64>, safety: Option<f64>) -> Result<f64, JsValue> {
    let c = c.unwrap_or(C0);
    let safety = safety.unwrap_or(0.99);
    if !(dx > 0.0) || !(dy > 0.0) || !(c > 0.0) {
        return Err(JsValue::from_str("courantDt requires positive dx, dy, and c"));
    }
    Ok(safety / (c * (1.0 / (dx * dx) + 1.0 / (dy * dy)).sqrt()))
}

fn sigma_norm_max(delta: f64) -> f64 {
    (0.8 * (PML_M as f64 + 1.0) * C0) / delta
}

fn sigma_at(x: f64, length: f64, thickness: f64, peak: f64) -> f64 {
    let mut depth = 0.0;
    if x < thickness {
        depth = thickness - x;
    } else if x > length - thickness {
        depth = x - (length - thickness);
    }
    if depth <= 0.0 || thickness <= 0.0 {
        return 0.0;
    }
    let rho = depth / thickness;
    peak * rho.powi(PML_M)
}

fn ab_of(sigma_norm: f64, dt: f64) -> (f64, f64) {
    if !(sigma_norm > 0.0) {
        return (0.0, 1.0);
    }
    let b = (-sigma_norm * dt).exp();
    (b - 1.0, b)
}

struct Axis {
    a_e: Vec<f64>,
    b_e: Vec<f64>,
    a_h: Vec<f64>,
    b_h: Vec<f64>,
    h0: usize,
    h1: usize,
    e0: usize,
    e1: isize,
}

fn fill_axis(n: usize, delta: f64, pml_cells: usize, dt: f64) -> Axis {
    let thickness = pml_cells as f64 * delta;
    let length = n as f64 * delta;
    let peak = sigma_norm_max(delta);
    let mut a_e = vec![0.0; n];
    let mut b_e = vec![0.0; n];
    let nh = n.saturating_sub(1);
    let mut a_h = vec![0.0; nh];
    let mut b_h = vec![0.0; nh];
    for i in 0..n {
        let (a, b) = ab_of(sigma_at((i as f64 + 0.5) * delta, length, thickness, peak), dt);
        a_e[i] = a;
        b_e[i] = b;
    }
    for i in 0..nh {
        let (a, b) = ab_of(sigma_at((i as f64 + 1.0) * delta, length, thickness, peak), dt);
        a_h[i] = a;
        b_h[i] = b;
    }
    let mut h0 = 0;
    while h0 < nh && a_h[h0] != 0.0 {
        h0 += 1;
    }
    let mut h1 = nh;
    while h1 > h0 && a_h[h1 - 1] != 0.0 {
        h1 -= 1;
    }
    let mut e0 = 0;
    while e0 < n && a_e[e0] != 0.0 {
        e0 += 1;
    }
    let mut e1 = n as isize - 1;
    while e1 >= e0 as isize && a_e[e1 as usize] != 0.0 {
        e1 -= 1;
    }
    Axis { a_e, b_e, a_h, b_h, h0, h1, e0, e1 }
}

struct Source {
    i: usize,
    j: usize,
    kind: u8, // 0 gaussian, 1 sine
    f0: f64,
    t0: f64,
    spread: f64,
    amp: f64,
    acc: f64,
}

struct Probe {
    i: usize,
    j: usize,
    samples: Vec<f64>,
}

#[allow(dead_code)]
struct Grid {
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    dt: f64,
    mode_tmz: bool,
    pml: usize,
    field: Vec<f64>,
    hx: Vec<f64>,
    hy: Vec<f64>,
    eps_r: Vec<f64>,
    sigma: Vec<f64>,
    pec: Vec<u8>,
    ca: Vec<f64>,
    cb: Vec<f64>,
    psi_ezx: Vec<f64>,
    psi_ezy: Vec<f64>,
    psi_hyx: Vec<f64>,
    psi_hxy: Vec<f64>,
    a_ex: Vec<f64>,
    b_ex: Vec<f64>,
    a_hx: Vec<f64>,
    b_hx: Vec<f64>,
    a_ey: Vec<f64>,
    b_ey: Vec<f64>,
    a_hy: Vec<f64>,
    b_hy: Vec<f64>,
    hy0: usize,
    hy1: usize,
    hx0: usize,
    hx1: usize,
    i_int0: isize,
    i_int1: isize,
    j_int0: isize,
    j_int1: isize,
    ch: f64,
    inv_dx: f64,
    inv_dy: f64,
    mu: f64,
    eps_base: f64,
    periodic_y: bool,
    sources: Vec<Source>,
    probes: Vec<Probe>,
    pec_list: Vec<usize>,
    time: f64,
    steps: u32,
}

impl Grid {
    fn eps_of(&self, eps_r: f64) -> f64 {
        self.eps_base * eps_r
    }

    fn write_coeff(&mut self, id: usize) {
        if self.pec[id] != 0 {
            self.ca[id] = 0.0;
            self.cb[id] = 0.0;
            return;
        }
        let eps = self.eps_of(self.eps_r[id]);
        let loss = (self.sigma[id] * self.dt) / (2.0 * eps);
        let den = 1.0 + loss;
        self.ca[id] = (1.0 - loss) / den;
        self.cb[id] = self.dt / eps / den;
    }

    fn rebuild_pec_list(&mut self) {
        self.pec_list.clear();
        for (id, cell) in self.pec.iter().enumerate() {
            if *cell != 0 {
                self.pec_list.push(id);
            }
        }
    }

    fn sample_source(src: &Source, t: f64) -> f64 {
        if src.kind == 1 {
            return src.amp * (2.0 * std::f64::consts::PI * src.f0 * t).sin();
        }
        if !(src.spread > 0.0) {
            return 0.0;
        }
        let x = (t - src.t0) / src.spread;
        let env = (-x * x).exp();
        if src.f0 != 0.0 {
            return src.amp * env * (2.0 * std::f64::consts::PI * src.f0 * (t - src.t0)).sin();
        }
        src.amp * env
    }

    fn update_hx(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let j_top = if self.periodic_y { ny } else { ny - 1 };
        let ch = self.ch;
        let inv_dy = self.inv_dy;
        let hx0 = self.hx0;
        let hx1 = self.hx1;
        let periodic_y = self.periodic_y;
        for j in 0..j_top {
            let up = if j + 1 < ny { j + 1 } else { 0 };
            let row = j * nx;
            let row_u = up * nx;
            let pml_row = !periodic_y && (j < hx0 || j >= hx1);
            if !pml_row {
                for i in 0..nx {
                    let id = row + i;
                    // row and row_u stay inside the field: up wraps only when periodic.
                    unsafe {
                        let dedy = (*self.field.get_unchecked(row_u + i) - *self.field.get_unchecked(id)) * inv_dy;
                        *self.hx.get_unchecked_mut(id) -= ch * dedy;
                    }
                }
                continue;
            }
            let a = self.a_hy[j];
            let b = self.b_hy[j];
            for i in 0..nx {
                let id = row + i;
                unsafe {
                    let dedy = (*self.field.get_unchecked(row_u + i) - *self.field.get_unchecked(id)) * inv_dy;
                    let psi = b * *self.psi_hxy.get_unchecked(id) + a * dedy;
                    *self.psi_hxy.get_unchecked_mut(id) = psi;
                    *self.hx.get_unchecked_mut(id) -= ch * (dedy + psi);
                }
            }
        }
    }

    fn update_hy(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let ch = self.ch;
        let inv_dx = self.inv_dx;
        let hy0 = self.hy0;
        let hy1 = self.hy1;
        for j in 0..ny {
            let row = j * nx;
            for i in 0..hy0 {
                let id = row + i;
                unsafe {
                    let dedx = (*self.field.get_unchecked(id + 1) - *self.field.get_unchecked(id)) * inv_dx;
                    let psi = *self.b_hx.get_unchecked(i) * *self.psi_hyx.get_unchecked(id)
                        + *self.a_hx.get_unchecked(i) * dedx;
                    *self.psi_hyx.get_unchecked_mut(id) = psi;
                    *self.hy.get_unchecked_mut(id) += ch * (dedx + psi);
                }
            }
            for i in hy0..hy1 {
                let id = row + i;
                unsafe {
                    let dedx = (*self.field.get_unchecked(id + 1) - *self.field.get_unchecked(id)) * inv_dx;
                    *self.hy.get_unchecked_mut(id) += ch * dedx;
                }
            }
            for i in hy1..nx - 1 {
                let id = row + i;
                unsafe {
                    let dedx = (*self.field.get_unchecked(id + 1) - *self.field.get_unchecked(id)) * inv_dx;
                    let psi = *self.b_hx.get_unchecked(i) * *self.psi_hyx.get_unchecked(id)
                        + *self.a_hx.get_unchecked(i) * dedx;
                    *self.psi_hyx.get_unchecked_mut(id) = psi;
                    *self.hy.get_unchecked_mut(id) += ch * (dedx + psi);
                }
            }
        }
    }

    fn update_ez_at(&mut self, i: usize, j: usize, with_psi: bool) {
        let nx = self.nx;
        let id = j * nx + i;
        if self.pec[id] != 0 {
            self.field[id] = 0.0;
            return;
        }
        let dhy = (self.hy[id] - self.hy[id - 1]) * self.inv_dx;
        let dhx = (self.hx[id] - self.hx[id - nx]) * self.inv_dy;
        if !with_psi {
            self.field[id] = self.ca[id] * self.field[id] + self.cb[id] * (dhy - dhx);
            return;
        }
        let px = self.b_ex[i] * self.psi_ezx[id] + self.a_ex[i] * dhy;
        let py = self.b_ey[j] * self.psi_ezy[id] + self.a_ey[j] * dhx;
        self.psi_ezx[id] = px;
        self.psi_ezy[id] = py;
        self.field[id] = self.ca[id] * self.field[id] + self.cb[id] * (dhy + px - (dhx + py));
    }

    fn update_ez_periodic(&mut self, i: usize, j: usize, below: usize, with_psi: bool) {
        let nx = self.nx;
        let id = j * nx + i;
        if self.pec[id] != 0 {
            self.field[id] = 0.0;
            return;
        }
        let dhy = (self.hy[id] - self.hy[id - 1]) * self.inv_dx;
        let dhx = (self.hx[id] - self.hx[below * nx + i]) * self.inv_dy;
        if !with_psi {
            self.field[id] = self.ca[id] * self.field[id] + self.cb[id] * (dhy - dhx);
            return;
        }
        let px = self.b_ex[i] * self.psi_ezx[id] + self.a_ex[i] * dhy;
        let py = self.b_ey[j] * self.psi_ezy[id] + self.a_ey[j] * dhx;
        self.psi_ezx[id] = px;
        self.psi_ezy[id] = py;
        self.field[id] = self.ca[id] * self.field[id] + self.cb[id] * (dhy + px - (dhx + py));
    }

    fn update_ez_interior(&mut self, j0: isize, j1: isize) {
        if j1 < j0 {
            return;
        }
        let nx = self.nx;
        let i0 = self.i_int0;
        let i1 = self.i_int1;
        let inv_dx = self.inv_dx;
        let inv_dy = self.inv_dy;
        for j in j0..=j1 {
            let row = j as usize * nx;
            let row_d = row - nx;
            for i in i0..=i1 {
                let id = row + i as usize;
                if self.pec[id] != 0 {
                    self.field[id] = 0.0;
                    continue;
                }
                unsafe {
                    let dhy = (*self.hy.get_unchecked(id) - *self.hy.get_unchecked(id - 1)) * inv_dx;
                    let dhx = (*self.hx.get_unchecked(id) - *self.hx.get_unchecked(row_d + i as usize)) * inv_dy;
                    let f = self.field.get_unchecked_mut(id);
                    *f = *self.ca.get_unchecked(id) * *f + *self.cb.get_unchecked(id) * (dhy - dhx);
                }
            }
        }
    }

    fn update_ez(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let i0 = self.i_int0;
        let i1 = self.i_int1;
        let j0 = self.j_int0;
        let j1 = self.j_int1;
        if !self.periodic_y {
            if j0 <= j1 && i0 <= i1 {
                self.update_ez_interior(j0, j1);
            }
            for j in 1..ny - 1 {
                let j = j as isize;
                let interior_row = j >= j0 && j <= j1;
                if interior_row {
                    let mut i = 1isize;
                    while i < i0 {
                        self.update_ez_at(i as usize, j as usize, true);
                        i += 1;
                    }
                    let mut i = i1 + 1;
                    while i < nx as isize - 1 {
                        self.update_ez_at(i as usize, j as usize, true);
                        i += 1;
                    }
                } else {
                    for i in 1..nx - 1 {
                        self.update_ez_at(i, j as usize, true);
                    }
                }
            }
            return;
        }
        for j in 0..ny {
            let below = (j + ny - 1) % ny;
            for i in 1..nx - 1 {
                let with_psi = (i as isize) < i0 || (i as isize) > i1;
                self.update_ez_periodic(i, j, below, with_psi);
            }
        }
    }

    fn step_n(&mut self, n: u32) {
        let nx = self.nx;
        for _ in 0..n {
            self.update_hx();
            self.update_hy();
            self.update_ez();
            self.time += self.dt;
            let t = self.time;
            for q in 0..self.sources.len() {
                let id = self.sources[q].j * nx + self.sources[q].i;
                if self.pec[id] != 0 {
                    continue;
                }
                let sample = Self::sample_source(&self.sources[q], t);
                self.sources[q].acc += sample;
                let acc = self.sources[q].acc;
                self.field[id] += acc;
            }
            let npec = self.pec_list.len();
            for p in 0..npec {
                let id = self.pec_list[p];
                self.field[id] = 0.0;
            }
            for q in 0..self.probes.len() {
                let id = self.probes[q].j * nx + self.probes[q].i;
                let v = self.field[id];
                self.probes[q].samples.push(v);
            }
            self.steps += 1;
        }
    }
}

#[wasm_bindgen]
pub struct WasmGrid {
    grid: Grid,
}

#[wasm_bindgen]
impl WasmGrid {
    pub fn dt(&self) -> f64 {
        self.grid.dt
    }
    pub fn nx(&self) -> u32 {
        self.grid.nx as u32
    }
    pub fn ny(&self) -> u32 {
        self.grid.ny as u32
    }
    pub fn steps(&self) -> u32 {
        self.grid.steps
    }
    pub fn field(&self) -> Vec<f64> {
        self.grid.field.clone()
    }
    pub fn step(&mut self, n: u32) {
        self.grid.step_n(n);
    }
    pub fn set_material(
        &mut self,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        eps_r: Option<f64>,
        sigma: Option<f64>,
        pec: Option<u8>,
    ) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let i0 = (x0.floor() as isize).clamp(0, nx as isize - 1) as usize;
        let i1 = (x1.floor() as isize).clamp(0, nx as isize - 1) as usize;
        let j0 = (y0.floor() as isize).clamp(0, ny as isize - 1) as usize;
        let j1 = (y1.floor() as isize).clamp(0, ny as isize - 1) as usize;
        let lo_i = i0.min(i1);
        let hi_i = i0.max(i1);
        let lo_j = j0.min(j1);
        let hi_j = j0.max(j1);
        for j in lo_j..=hi_j {
            for i in lo_i..=hi_i {
                let id = j * nx + i;
                if let Some(e) = eps_r {
                    self.grid.eps_r[id] = e;
                }
                if let Some(s) = sigma {
                    self.grid.sigma[id] = s;
                }
                if let Some(p) = pec {
                    self.grid.pec[id] = if p != 0 { 1 } else { 0 };
                }
                self.grid.write_coeff(id);
            }
        }
        if pec.is_some() {
            self.grid.rebuild_pec_list();
        }
    }
    pub fn add_source(
        &mut self,
        i: i32,
        j: i32,
        kind: &str,
        f0: f64,
        t0: f64,
        spread: f64,
        amp: f64,
    ) -> Result<u32, JsValue> {
        let i = i as isize;
        let j = j as isize;
        if i < 0 || j < 0 || i >= self.grid.nx as isize || j >= self.grid.ny as isize {
            return Err(JsValue::from_str(&format!("source ({i},{j}) is outside the grid")));
        }
        let kind = match kind {
            "gaussian" => 0,
            "sine" => 1,
            other => return Err(JsValue::from_str(&format!("unknown source kind {other}"))),
        };
        self.grid.sources.push(Source {
            i: i as usize,
            j: j as usize,
            kind,
            f0,
            t0,
            spread,
            amp,
            acc: 0.0,
        });
        Ok((self.grid.sources.len() - 1) as u32)
    }
    pub fn add_probe(&mut self, i: i32, j: i32) -> Result<u32, JsValue> {
        let i = i as isize;
        let j = j as isize;
        if i < 0 || j < 0 || i >= self.grid.nx as isize || j >= self.grid.ny as isize {
            return Err(JsValue::from_str(&format!("probe ({i},{j}) is outside the grid")));
        }
        self.grid.probes.push(Probe {
            i: i as usize,
            j: j as usize,
            samples: Vec::new(),
        });
        Ok((self.grid.probes.len() - 1) as u32)
    }
    pub fn probe_len(&self) -> u32 {
        self.grid.probes.len() as u32
    }
    pub fn probe_samples(&self, id: u32) -> Result<Vec<f64>, JsValue> {
        self.grid
            .probes
            .get(id as usize)
            .map(|p| p.samples.clone())
            .ok_or_else(|| JsValue::from_str("probe id out of range"))
    }
}

#[wasm_bindgen]
pub fn create_grid(
    nx: u32,
    ny: u32,
    dx: f64,
    dy: f64,
    mode: &str,
    pml: u32,
    periodic_y: bool,
) -> Result<WasmGrid, JsValue> {
    let mode_tmz = match mode {
        "tmz" => true,
        "scalar" => false,
        other => return Err(JsValue::from_str(&format!("mode must be 'tmz' or 'scalar', got {other}"))),
    };
    let nx = nx as usize;
    let ny = ny as usize;
    let min_ny = if periodic_y { 2 } else { 3 };
    if nx < 3 || ny < min_ny {
        return Err(JsValue::from_str(&format!(
            "nx and ny must be integers (nx >= 3, ny >= {min_ny})"
        )));
    }
    if !(dx > 0.0) || !(dy > 0.0) {
        return Err(JsValue::from_str("dx and dy must be positive"));
    }
    let pml_cells = pml as usize;
    if pml_cells * 2 >= nx || (!periodic_y && pml_cells * 2 >= ny) {
        return Err(JsValue::from_str("pml thickness leaves no interior"));
    }
    let dt = courant_dt(dx, dy, None, None)?;
    let n = nx * ny;
    let x_axis = fill_axis(nx, dx, pml_cells, dt);
    let y_axis = fill_axis(ny, dy, if periodic_y { 0 } else { pml_cells }, dt);
    let mu = if mode_tmz { MU0 } else { 1.0 };
    let eps_base = if mode_tmz { EPS0 } else { 1.0 / (C0 * C0) };
    let mut grid = Grid {
        nx,
        ny,
        dx,
        dy,
        dt,
        mode_tmz,
        pml: pml_cells,
        field: vec![0.0; n],
        hx: vec![0.0; n],
        hy: vec![0.0; n],
        eps_r: vec![1.0; n],
        sigma: vec![0.0; n],
        pec: vec![0; n],
        ca: vec![0.0; n],
        cb: vec![0.0; n],
        psi_ezx: vec![0.0; n],
        psi_ezy: vec![0.0; n],
        psi_hyx: vec![0.0; n],
        psi_hxy: vec![0.0; n],
        a_ex: x_axis.a_e,
        b_ex: x_axis.b_e,
        a_hx: x_axis.a_h,
        b_hx: x_axis.b_h,
        a_ey: y_axis.a_e,
        b_ey: y_axis.b_e,
        a_hy: y_axis.a_h,
        b_hy: y_axis.b_h,
        hy0: x_axis.h0,
        hy1: x_axis.h1,
        hx0: y_axis.h0,
        hx1: y_axis.h1,
        i_int0: 1.max(x_axis.e0 as isize),
        i_int1: (nx as isize - 2).min(x_axis.e1),
        j_int0: 1.max(y_axis.e0 as isize),
        j_int1: (ny as isize - 2).min(y_axis.e1),
        ch: dt / mu,
        inv_dx: 1.0 / dx,
        inv_dy: 1.0 / dy,
        mu,
        eps_base,
        periodic_y,
        sources: Vec::new(),
        probes: Vec::new(),
        pec_list: Vec::new(),
        time: 0.0,
        steps: 0,
    };
    for id in 0..n {
        grid.write_coeff(id);
    }
    Ok(WasmGrid { grid })
}
