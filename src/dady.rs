use crate::linealg::{Tensor, Vector};

pub trait Function {
    fn input(&self) -> Option<&dyn Function>;

    fn input_mut(&mut self) -> Option<&mut dyn Function>;
    
    fn params(&self) -> &Tensor;

    fn params_mut(&mut self) -> &mut Tensor;

    /// Caculates output.
    fn forward(&self) -> Tensor;

    /// Caculates gradient.
    /// Returns (Gradient caused by params, Gradient caused by input)
    fn gradient(&self) -> (Tensor, Tensor);
}

pub struct Constant(pub Tensor);

impl Function for Constant {
    fn forward(&self) -> Tensor {
        self.0.clone()
    }
    fn gradient(&self) -> (Tensor, Tensor) {
        (Tensor::S(1.0), Tensor::S(0.0))
    }
    fn params_mut(&mut self) -> &mut Tensor {
        &mut self.0
    }
    fn params(&self) -> &Tensor {
        &self.0
    }
    fn input(&self) -> Option<&dyn Function> {
        None
    }
    fn input_mut(&mut self) -> Option<&mut dyn Function> {
        None
    }
}

pub struct Polynomial {
    param: Tensor,
    input: Box<dyn Function>,
}

impl Polynomial {
    pub fn new(param: Vector, input: Box<dyn Function>) -> Self {
        Self { param: Tensor::V(param), input }
    }
}

impl Function for Polynomial {
    fn forward(&self) -> Tensor {
        let param = self.param.as_vector().unwrap();
        let input = self.input.forward().as_scalar().unwrap();
        let mut sum = 0.0;
        for i in 0..param.size() {
            sum += param.0[i] * input.powi(i as i32);
        }
        Tensor::S(sum)
    }

    fn gradient(&self) -> (Tensor, Tensor) {
        let param = self.param.as_vector().unwrap();
        let input = self.input.forward().as_scalar().unwrap();
        let mut param_grad = vec![0.0; param.size()];
        for i in 0..param.size() {
            param_grad[i] = input.powi(i as i32);
        }
        let mut input_grad = 0.0;
        for i in 0..param.size() {
            let a = param.0[i] * i as f32;
            let b = input.powi((i + 1) as i32);
            input_grad += a * b;
        }
        (Tensor::V(Vector(param_grad)), Tensor::S(input_grad))
    }

    fn params_mut(&mut self) -> &mut Tensor {
        &mut self.param
    }

    fn params(&self) -> &Tensor {
        &self.param
    }

    fn input(&self) -> Option<&dyn Function> {
        Some(self.input.as_ref())
    }

    fn input_mut(&mut self) -> Option<&mut dyn Function> {
        Some(self.input.as_mut())
    }
}

pub struct SqLoss {
    input: Box<dyn Function>,
    target: Tensor,
}

impl SqLoss {
    pub fn new(input: Box<dyn Function>, target: Tensor) -> Self {
        Self { input, target }
    }
}

impl Function for SqLoss {
    fn forward(&self) -> Tensor {
        let x = self.input.forward().as_vector().unwrap();
        let y = self.target.as_vector().unwrap();
        Tensor::S((x - y).len_sq() * 0.5)
    }
    fn gradient(&self) -> (Tensor, Tensor) {
        let x = self.input.forward().as_vector().unwrap();
        let y = self.target.as_vector().unwrap();
        (Tensor::V(y.clone() - x.clone()), Tensor::V(x - y))
    }
    
    fn input(&self) -> Option<&dyn Function> {
        Some(self.input.as_ref())
    }
    
    fn input_mut(&mut self) -> Option<&mut dyn Function> {
        Some(self.input.as_mut())
    }
    
    fn params(&self) -> &Tensor {
        &self.target
    }
    
    fn params_mut(&mut self) -> &mut Tensor {
        &mut self.target
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_polynomial_call() {
//         let coeffs = Tensor(vec![1.0, 2.0, 3.0]);
//         let poly = Polynomial::new(coeffs);

//         assert_eq!(poly.call(&Tensor(vec![0.0])), Tensor(vec![1.0]));
//         assert_eq!(poly.call(&Tensor(vec![1.0])), Tensor(vec![6.0]));
//         assert_eq!(poly.call(&Tensor(vec![2.0])), Tensor(vec![17.0]));
//     }

//     #[test]
//     fn test_polynomial_gradient() {
//         let coeffs = Tensor(vec![1.0, 2.0, 3.0]);
//         let poly = Polynomial::new(coeffs);
//         assert_eq!(poly.gradient(&Tensor(vec![0.0])).0, Tensor(vec![1.0, 0.0, 0.0]));
//         assert_eq!(poly.gradient(&Tensor(vec![1.0])).0, Tensor(vec![1.0, 1.0, 1.0]));
//         assert_eq!(poly.gradient(&Tensor(vec![2.0])).0, Tensor(vec![1.0, 2.0, 4.0]));
//     }
// }
