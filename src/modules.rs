use std::{iter::zip, vec};

use rand::distributions::{Bernoulli, Distribution};

use crate::linealg::{Matrix, Tensor, Vector};

pub trait Function {
    /// Caculates output.
    fn evaluate(&self, param: &Tensor, input: &Vector) -> Vector;

    /// Evaluate and prepare for backward.
    fn forward(&mut self, param: &Tensor, input: &Vector) -> Vector {
        self.evaluate(param, input)
    }

    /// Returns (gradient of param, new delta)
    fn backward(&self, param: &Tensor, input: &Vector, delta: Vector) -> (Tensor, Vector);
}

pub struct Linear;

impl Function for Linear {
    fn evaluate(&self, param: &Tensor, input: &Vector) -> Vector {
        let matrix = param.clone().as_matrix();
        matrix.apply(&input)
    }
    fn backward(&self, param: &Tensor, input: &Vector, delta: Vector) -> (Tensor, Vector) {
        let mut param_grad: Vec<Vector> = vec![];
        for a in input.0.iter() {
            let mut grad: Vec<f32> = vec![];
            for d in delta.0.iter() {
                grad.push(d * a);
            }
            param_grad.push(Vector(grad));
        }
        let param_grad = Matrix::new(param_grad);

        let matrix = param.clone().as_matrix();
        let matrix_t = matrix.transpose();
        assert_eq!(param_grad.shape(), matrix.shape());
        let delta = matrix_t.apply(&delta);
        (Tensor::M(param_grad), delta)
    }
}

pub struct Translation;

impl Function for Translation {
    fn evaluate(&self, param: &Tensor, input: &Vector) -> Vector {
        let param = param.clone().as_vector();
        param + input
    }
    fn backward(&self, _param: &Tensor, _input: &Vector, delta: Vector) -> (Tensor, Vector) {
        (Tensor::V(delta.clone()), delta)
    }
}

pub struct Dropout {
    p: f32,
    droped: Vec<bool>,
}

impl Dropout {
    pub fn new(p: f32, len: usize) -> Self {
        Self {
            p,
            droped: vec![false; len],
        }
    }
}

impl Function for Dropout {
    fn evaluate(&self, _param: &Tensor, input: &Vector) -> Vector {
        input.clone()
    }

    fn forward(&mut self, _param: &Tensor, input: &Vector) -> Vector {
        let mut v = input.0.clone();
        let dist = Bernoulli::new(self.p as f64).unwrap();
        for i in 0..v.len() {
            if dist.sample(&mut rand::thread_rng()) {
                v[i] = 0.0;
                self.droped[i] = true;
            } else {
                v[i] /= 1.0 - self.p;
                self.droped[i] = false;
            }
        }
        Vector(v)
    }
    fn backward(&self, _param: &Tensor, _input: &Vector, delta: Vector) -> (Tensor, Vector) {
        let mut v = delta.0;
        for i in 0..v.len() {
            if self.droped[i] {
                v[i] = 0.0;
            } else {
                v[i] /= 1.0 - self.p;
            }
        }
        (Tensor::N, Vector(v))
    }
}

trait ActivationFunction {
    fn eval(&self, input: f32) -> f32;

    fn derivetive(&self, input: f32) -> f32;
}

impl<T> Function for T
where
    T: ActivationFunction,
{
    fn evaluate(&self, _param: &Tensor, input: &Vector) -> Vector {
        Vector(input.0.iter().map(|x| self.eval(*x)).collect())
    }

    fn backward(&self, _param: &Tensor, input: &Vector, delta: Vector) -> (Tensor, Vector) {
        let derivatives = Vector(input.0.iter().map(|x| self.derivetive(*x)).collect());

        (Tensor::N, derivatives.hadamard(&delta))
    }
}

pub struct ReLU;

impl ActivationFunction for ReLU {
    fn eval(&self, input: f32) -> f32 {
        input.max(0.0)
    }
    fn derivetive(&self, input: f32) -> f32 {
        if input >= 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

pub struct ELU {
    pub alpha: f32,
}

impl ActivationFunction for ELU {
    fn eval(&self, input: f32) -> f32 {
        if input >= 0.0 {
            input
        } else {
            self.alpha * (input.exp() - 1.0)
        }
    }
    fn derivetive(&self, input: f32) -> f32 {
        if input >= 0.0 {
            1.0
        } else {
            self.alpha * input.exp()
        }
    }
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn eval(&self, input: f32) -> f32 {
        1.0 / (1.0 + (-input).exp())
    }
    fn derivetive(&self, input: f32) -> f32 {
        (1. + (-input).exp()).powi(-2) * (-input).exp()
    }
}

pub trait LossFunction {
    fn loss(&self, target: &Vector, input: &Vector) -> f32;

    fn delta(&self, target: &Vector, input: &Vector) -> Vector;
}

pub struct MSELoss;

impl LossFunction for MSELoss {
    fn loss(&self, target: &Vector, input: &Vector) -> f32 {
        (input.clone() - target).len_sq() * 0.5
    }
    fn delta(&self, target: &Vector, input: &Vector) -> Vector {
        input.clone() - target
    }
}

pub struct CrossEntropyLoss;

impl LossFunction for CrossEntropyLoss {
    fn loss(&self, target: &Vector, input: &Vector) -> f32 {
        let mut sum = 0.0;
        for i in 0..input.size() {
            let x = input.0[i];
            let y = target.0[i];
            debug_assert!(
                (0.0..=1.0).contains(&x),
                "input have to be in [0, 1] but got {}",
                x
            );
            debug_assert!(
                (0.0..=1.0).contains(&y),
                "target have to be in [0, 1] but got {}",
                y
            );
            let x = x.clamp(1e-5, 1. - 1e-5);
            let y = y.clamp(1e-5, 1. - 1e-5);
            let loss = -y * x.ln() - (1. - y) * (1. - x).ln();
            sum += loss;
        }
        debug_assert!(sum >= 0.0);
        sum
    }

    fn delta(&self, target: &Vector, input: &Vector) -> Vector {
        let mut delta = vec![];
        for i in 0..input.size() {
            let x = input.0[i];
            let y = target.0[i];
            let x = x.clamp(1e-5, 1. - 1e-5);
            let y = y.clamp(1e-5, 1. - 1e-5);
            delta.push(-y / x + (1. - y) / (1. - x));
        }
        Vector(delta)
    }
}

pub struct Sequence {
    funcs: Vec<Box<dyn Function>>,
    value_cache: Vec<Vector>,
}

impl Sequence {
    pub fn new(funcs: Vec<Box<dyn Function>>) -> Self {
        Self {
            funcs,
            value_cache: vec![],
        }
    }
    
}

impl Function for Sequence {
    fn evaluate(&self, param: &Tensor, input: &Vector) -> Vector {
        let mut input = input.clone();
        let params = param.as_tensor_list_ref();
        for (func, param) in zip(&self.funcs, params) {
            input = func.evaluate(param, &input);
        }
        input
    }

    fn forward(&mut self, param: &Tensor, input: &Vector) -> Vector {
        self.value_cache.clear();
        let mut input = input.clone();
        let params = param.as_tensor_list_ref();

        for (func, param) in zip(&mut self.funcs, params) {
            let output = func.forward(param, &input);
            self.value_cache.push(output.clone());
            input = output;
        }

        input
    }

    fn backward(&self, param: &Tensor, input: &Vector, mut delta: Vector) -> (Tensor, Vector) {
        let mut param_grads = vec![];
        let params = param.as_tensor_list_ref();
        for (i, func) in self.funcs.iter().enumerate().rev() {
            let param = &params[i];
            let input = if i == 0 {
                input
            } else {
                &self.value_cache[i - 1]
            };
            let (grad, new_delta) = func.backward(param, input, delta);
            delta = new_delta;
            param_grads.insert(0, grad);
        }

        (Tensor::L(param_grads), delta)
    }
}
