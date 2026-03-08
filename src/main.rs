use std::{fs::File, io::Write, time::Instant};

const DBG: bool = true;

const G: f64 = 0.1;
const EPSILON: f64 = 0.01;

const M: f64 = 10.0;
const Q: f64 = 2.0;
const A: f64 = 4.0;
const OMEGA: f64 = 0.1;
const SIZE: f64 = 8.0;
const GAMMA: f64 = 0.01 * OMEGA;


const M1: f64 = M/(1.0 + Q);
const M2: f64 = M - M1;
const A1: f64 = (Q * A)/(Q + 1.0);
const A2: f64 = - A1/Q;

fn dot(u: (f64, f64), v: (f64, f64)) -> f64 {
    return u.0 * v.0 + u.1 * v.1;
}

fn roche_potential(x: f64, y: f64) -> f64 {
    let r = (x.powi(2) + y.powi(2)).sqrt();
    let r1 = ((x - A1).powi(2) + y.powi(2) + EPSILON.powi(2)).sqrt();
    let r2 = ((x - A2).powi(2) + y.powi(2) + EPSILON.powi(2)).sqrt();
    let phi1 = - (G * M1)/(r1);
    let phi2 = - (G * M2)/(r2);
    let phi_c = (0.5) * (OMEGA.powi(2)) * (r.powi(2));
    let phi = phi1 + phi2 - phi_c;
    return phi;
}

fn grad_roche(x: f64, y: f64) -> (f64, f64) {
    let r1 = ((x - A1).powi(2) + y.powi(2) + EPSILON.powi(2)).sqrt();
    let r2 = ((x - A2).powi(2) + y.powi(2) + EPSILON.powi(2)).sqrt();
    return ((- (G * M1 * (x - A1))/r1.powi(3) - (G * M2 * (x - A2))/r2.powi(3) + (OMEGA.powi(2)) * x), (- (G * M1 * y)/r1.powi(3) - (G * M2 * y)/r2.powi(3) + (OMEGA.powi(2)) * y))
}

fn grad_roche_x(x: f64, y: f64) -> f64 {
    return grad_roche(x, y).0;
}

fn grad_roche_y(x: f64, y: f64) -> f64 {
    return grad_roche(x, y).1;
}

fn acc(x: f64, y: f64, vx: f64, vy: f64) -> (f64, f64) {
    let rel_x = x - A1;
    let rel_y = y;
    let v = (vx, vy);
    let r_norm = (rel_x.powi(2) + rel_y.powi(2)).sqrt();
    let r_normalized = (rel_x/r_norm, rel_y/r_norm);
    let v_rad = dot(v, r_normalized);
    let a_diss_x = - GAMMA * v_rad * r_normalized.0;
    let a_diss_y = - GAMMA * v_rad * r_normalized.1;
    let grad_roche_value = grad_roche(x, y);
    return (grad_roche_value.0 + 2.0 * OMEGA * vy + a_diss_x, grad_roche_value.1 - 2.0 * OMEGA * vx + a_diss_y);
}

fn test_particule_rk4_adaptative(x0: f64, y0: f64, vx0: f64, vy0: f64, step: usize, dt0: f64, sample_size: usize) -> (Vec<(f64, f64)>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let now = Instant::now();

    let save_step = step/sample_size;

    let mut total_time = 0.0;
    let mut x = x0;
    let mut y = y0;
    let mut vx = vx0;
    let mut vy = vy0;
    let mut rel_x = x - A1;
    let mut rel_y = y;
    let norm0 = (vx0.powi(2) + vy0.powi(2) + EPSILON.powi(2)).sqrt();
    let mut norm = (vx.powi(2) + vy.powi(2) + EPSILON.powi(2)).sqrt();
    let mut pos_list: Vec<(f64, f64)> = Vec::with_capacity(sample_size);
    pos_list.push((x0, y0));
    let mut jacobi_cst: Vec<f64> = Vec::with_capacity(sample_size);
    jacobi_cst.push((-2.0) * roche_potential(x, y) - (vx.powi(2) + vy.powi(2)));
    let r_norm = (rel_x.powi(2) + rel_y.powi(2)).sqrt();
    let mut radius_list = Vec::with_capacity(sample_size);
    radius_list.push(r_norm);
    let mut time_list = Vec::with_capacity(sample_size);
    time_list.push(total_time);
    for current_step in 0..step {
        let dt = dt0 * (norm0/norm);
        // let k1x = grad_roche_x(x, y) + 2.0 * OMEGA * vy;
        // let k1y = grad_roche_y(x, y) - 2.0 * OMEGA * vx;
        // let k2x = grad_roche_x(x + (dt/2.0) * vx, y + (dt/2.0) * vy) + 2.0 * OMEGA * (vy + (dt/2.0) * k1y);
        // let k2y = grad_roche_y(x + (dt/2.0) * vx, y + (dt/2.0) * vy) - 2.0 * OMEGA * (vx + (dt/2.0) * k1x);
        // let k3x = grad_roche_x(x + (dt/2.0) * vx + ((dt.powi(2))/4.0) * k1x, y + (dt/2.0) * vy + ((dt.powi(2))/4.0) * k1y) + 2.0 * OMEGA * (vy + (dt/2.0) * k2y);
        // let k3y = grad_roche_y(x + (dt/2.0) * vx + ((dt.powi(2))/4.0) * k1x, y + (dt/2.0) * vy + ((dt.powi(2))/4.0) * k1y) - 2.0 * OMEGA * (vx + (dt/2.0) * k2x);
        // let k4x = grad_roche_x(x + dt * vx + ((dt.powi(2))/2.0) * k2x, y + dt * vy + ((dt.powi(2))/2.0) * k2y) + 2.0 * OMEGA * (vy + dt * k3y);
        // let k4y = grad_roche_y(x + dt * vx + ((dt.powi(2))/2.0) * k2x, y + dt * vy + ((dt.powi(2))/2.0) * k2y) - 2.0 * OMEGA * (vx + dt * k3x);

        let k1 = acc(x, y, vx, vy);
        let k1x = k1.0;
        let k1y = k1.1;
        let k2 = acc(x + (dt/2.0) * vx, y + (dt/2.0) * vy, vx + (dt/2.0) * k1x, vy + (dt/2.0) * k1y);
        let k2x = k2.0;
        let k2y = k2.1;
        let k3 = acc(x + (dt/2.0) * vx + ((dt.powi(2))/4.0) * k1x, y + (dt/2.0) * vy + ((dt.powi(2))/4.0) * k1y, vx + (dt/2.0) * k2x, vy + (dt/2.0) * k2y);
        let k3x = k3.0;
        let k3y = k3.1;
        let k4 = acc(x + dt * vx + ((dt.powi(2))/2.0) * k2x, y + dt * vy + ((dt.powi(2))/2.0) * k2y, vx + dt * k3x, vy + dt * k3y);
        let k4x = k4.0;
        let k4y = k4.1;

        x = x + dt * vx + ((dt.powi(2))/6.0) * (k1x + k2x + k3x);
        y = y + dt * vy + ((dt.powi(2))/6.0) * (k1y + k2y + k3y);
        vx = vx + (dt/6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        vy = vy + (dt/6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        norm = (vx.powi(2) + vy.powi(2) + EPSILON.powi(2)).sqrt();
        if (x.powi(2) + y.powi(2)).sqrt() > SIZE {
            if DBG {println!("Distance is too large, stop")}
            break
        }

        total_time += dt;
        rel_x = x - A1;
        rel_y = y;

        if (current_step + 1) % save_step == 0 { // We save the step in vec
            pos_list.push((x, y));
            jacobi_cst.push((-2.0) * roche_potential(x, y) - (vx.powi(2) + vy.powi(2)));
            let r_norm = (rel_x.powi(2) + rel_y.powi(2)).sqrt();
            radius_list.push(r_norm);
            time_list.push(total_time);
        }
    }
    let elapsed = now.elapsed();
    println!("{:.2?}", elapsed);
    return (pos_list, jacobi_cst, radius_list, time_list);
}

fn main() {
    let x_l1 = 0.9706573406008301;
    let (pos_list, jacobi_cst, radius_list, time_list) = test_particule_rk4_adaptative(x_l1, - 0.001 * A, 0.0, 0.0, 1000000000, 0.01, 10000000);
    let mut file = File::create("pos.txt").unwrap();
    let mut content = String::new();
    // let sample_size = 100000;
    // let step = pos_list.len()/sample_size;
    // for i in 0..sample_size {
    //     let index = i * step;
    //     let pos = pos_list[index];
    //     let jacobi = jacobi_cst[index];
    //     let radius = radius_list[index];
    //     let time = time_list[index];
    //     content.push_str(&format!("{} {} {} {} {}\n", pos.0, pos.1, jacobi, radius, time));
    // }
    for i in 0..pos_list.len() {
        let pos = pos_list[i];
        let jacobi = jacobi_cst[i];
        let radius = radius_list[i];
        let time = time_list[i];
        content.push_str(&format!("{} {} {} {} {}\n", pos.0, pos.1, jacobi, radius, time));
    }
    file.write_all(content.as_bytes()).unwrap();
}
