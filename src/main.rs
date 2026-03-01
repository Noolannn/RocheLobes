use std::{fs::File, io::Write, time::Instant};

const DBG: bool = true;

const G: f64 = 0.1;
const EPSILON: f64 = 0.01;

const M: f64 = 10.0;
const Q: f64 = 2.0;
const A: f64 = 4.0;
const OMEGA: f64 = 0.1;
const SIZE: f64 = 8.0;


const M1: f64 = M/(1.0 + Q);
const M2: f64 = M - M1;
const A1: f64 = (Q * A)/(Q + 1.0);
const A2: f64 = - A1/Q;

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

fn test_particule_rk4_adaptative(x0: f64, y0: f64, vx0: f64, vy0: f64, step: usize, dt0: f64) -> (Vec<(f64, f64)>, Vec<f64>) {
    let now = Instant::now();

    let mut x = x0;
    let mut y = y0;
    let mut vx = vx0;
    let mut vy = vy0;
    let norm0 = (vx0.powi(2) + vy0.powi(2) + EPSILON.powi(2)).sqrt();
    let mut norm = (vx.powi(2) + vy.powi(2) + EPSILON.powi(2)).sqrt();
    let mut pos_list: Vec<(f64, f64)> = vec![(x0, y0)];
    let mut jacobi_cst: Vec<f64> = vec![(-2.0) * roche_potential(x, y) - (vx.powi(2) + vy.powi(2))];
    for _ in 0..step {
        let dt = dt0 * (norm0/norm);
        let k1x = grad_roche_x(x, y) + 2.0 * OMEGA * vy;
        let k1y = grad_roche_y(x, y) - 2.0 * OMEGA * vx;
        let k2x = grad_roche_x(x + (dt/2.0) * vx, y + (dt/2.0) * vy) + 2.0 * OMEGA * (vy + (dt/2.0) * k1y);
        let k2y = grad_roche_y(x + (dt/2.0) * vx, y + (dt/2.0) * vy) - 2.0 * OMEGA * (vx + (dt/2.0) * k1x);
        let k3x = grad_roche_x(x + (dt/2.0) * vx + ((dt.powi(2))/4.0) * k1x, y + (dt/2.0) * vy + ((dt.powi(2))/4.0) * k1y) + 2.0 * OMEGA * (vy + (dt/2.0) * k2y);
        let k3y = grad_roche_y(x + (dt/2.0) * vx + ((dt.powi(2))/4.0) * k1x, y + (dt/2.0) * vy + ((dt.powi(2))/4.0) * k1y) - 2.0 * OMEGA * (vx + (dt/2.0) * k2x);
        let k4x = grad_roche_x(x + dt * vx + ((dt.powi(2))/2.0) * k2x, y + dt * vy + ((dt.powi(2))/2.0) * k2y) + 2.0 * OMEGA * (vy + dt * k3y);
        let k4y = grad_roche_y(x + dt * vx + ((dt.powi(2))/2.0) * k2x, y + dt * vy + ((dt.powi(2))/2.0) * k2y) - 2.0 * OMEGA * (vx + dt * k3x);

        x = x + dt * vx + ((dt.powi(2))/6.0) * (k1x + k2x + k3x);
        y = y + dt * vy + ((dt.powi(2))/6.0) * (k1y + k2y + k3y);
        vx = vx + (dt/6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        vy = vy + (dt/6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        norm = (vx.powi(2) + vy.powi(2) + EPSILON.powi(2)).sqrt();
        if (x.powi(2) + y.powi(2)).sqrt() > SIZE {
            if DBG {println!("Distance is too large, stop")}
            break
        }
        pos_list.push((x, y));
        jacobi_cst.push((-2.0) * roche_potential(x, y) - (vx.powi(2) + vy.powi(2)));
    }
    let elapsed = now.elapsed();
    println!("{:.2?}", elapsed);
    return (pos_list, jacobi_cst);
}

fn main() {
    let x_l1 = 0.9706573406008301;
    let (pos_list, jacobi_cst) = test_particule_rk4_adaptative(x_l1, - 0.001 * A, 0.0, 0.0, 1000000, 0.01);
    let mut file = File::create("pos.txt").unwrap();
    let mut content = String::new();
    for pos in pos_list {
        content.push_str(&format!("{} {}\n", pos.0, pos.1));
    }
    file.write_all(content.as_bytes()).unwrap();
}
