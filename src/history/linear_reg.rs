use dady::SqLoss;
use rand::{thread_rng, Rng};

//mod nn;
mod dady;
mod linealg;

use crate::linealg::Vector;

pub struct FirstOrder {
    pub k: f32,
    pub b: f32,
}

impl FirstOrder {
    pub fn call(&self, x: f32) -> f32 {
        self.k * x + self.b
    }

    pub fn gradient(&self, x: f32) -> Vector {
        // [del k, del b, del x]
        Vector(vec![x, 1.0, self.k])
    }

    pub fn shape(&self) -> (usize, usize) {
        // 2 params, 1 input
        (2, 1)
    }
}

pub struct SqLoss;

impl SqLoss {
    pub fn call(&self, x: f32, y: f32) -> f32 {
        (x - y).powi(2)
    }
    pub fn gradient(&self, x: f32, y: f32) -> Vector {
        // [del x, del y]
        Vector(vec![2. * (x - y), 2. * (y - x)])
    }
}

fn fntoapproch(x: f32) -> f32 {
    2.0 * x + 1.
}

fn main() {
    let mut fs = FirstOrder { k: -300., b: -100. };
    let loss = SqLoss;
    let lr = 0.01;
    for i in 0..1000 {
        let x = thread_rng().gen_range(-10.0..10.);
        if i % 100 == 0 {
            println!(
                "iter {i}, f({}) = {}, loss = {}",
                x,
                fs.call(x),
                loss.call(fs.call(x), fntoapproch(x))
            );  
        }
        let y = fntoapproch(x);

        let c_del_f = loss.gradient(fs.call(x), y).0[0];
        let f_del_k = fs.gradient(x).0[0];
        let f_del_b = fs.gradient(x).0[1];
        fs.k -= lr * f_del_k * c_del_f;
        fs.b -= lr * f_del_b * c_del_f;
    }
    println!("k = {}, b = {}", fs.k, fs.b);
}
