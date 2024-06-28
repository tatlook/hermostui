use linealg::Vector;
use rand::{thread_rng, Rng};

pub struct Polynomial {
    pub coeffs: Vector,
}

impl Polynomial {
    pub fn new(coeffs: Vector) -> Self {
        Self { coeffs }
    }
}

impl Polynomial {
    pub fn call(&self, x: f32) -> f32 {
        let mut sum = 0.0;
        for i in 0..self.coeffs.size() {
            sum += self.coeffs.0[i] * x.powi(i as i32);
        }
        sum
    }

    pub fn gradient(&self, x: f32) -> (Vector, Vector) {
        let mut grad = vec![0.0; self.coeffs.size()];
        for i in 0..self.coeffs.size() {
            grad[i] = x.powi(i as i32);
        }
        (Vector(grad), Vector(vec![0.0]))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_call() {
        let coeffs = Vector(vec![1.0, 2.0, 3.0]);
        let poly = Polynomial::new(coeffs);

        assert_eq!(poly.call(0.0), 1.0);
        assert_eq!(poly.call(1.0), 6.0);
        assert_eq!(poly.call(2.0), 17.0);
    }
    #[test]
    fn test_polynomial_gradient() {
        let coeffs = Vector(vec![1.0, 2.0, 3.0]);
        let poly = Polynomial::new(coeffs);

        assert_eq!(poly.gradient(0.0).0, Vector(vec![1.0, 0.0, 0.0]));
        assert_eq!(poly.gradient(1.0).0, Vector(vec![1.0, 1.0, 1.0]));
        assert_eq!(poly.gradient(2.0).0, Vector(vec![1.0, 2.0, 4.0]));
    }
}

mod linealg;

fn fntoapproch(x: f32) -> f32 {
    3. + 2.0 * x + x.powi(2)
}

fn main() {
    let mut fs = Polynomial::new(Vector(vec![3., -1.99999, 1.0]));
    let loss = SqLoss;
    let lr = 0.001;
    let batch_size = 500;
    for _ in 0..1000000 / batch_size {
        let mut nabla_c = Vector(vec![0., 0., 0.]);
        let x = thread_rng().gen_range(5.0..5.5);
        let y = fntoapproch(x);
        for _ in 0..batch_size {

            let c_del_f = loss.gradient(fs.call(x), y).0[0];
            let (nabla_f, _) = fs.gradient(x);
            let nabla_c_k = nabla_f.clone() * c_del_f;
            nabla_c += nabla_c_k * (1. / batch_size as f32);
        }
        fs.coeffs -= nabla_c.clone() * lr;
    }
    println!("{}", fs.coeffs);
}

#[cfg(test)]
mod test {
    #[test]
    fn reduce() {
        let v = vec![1., 2., 4., 5.];
        assert_eq!(Vec::from(&v[..]), v);
        assert_eq!(Vec::from(&v[..v.len()]), v);
        assert_eq!(Vec::from(&v[..v.len() - 1]), vec![1., 2., 4.]);
    }
}
