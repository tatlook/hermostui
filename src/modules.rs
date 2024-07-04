use std::vec;

use crate::linealg::{Matrix, Tensor, Vector};

pub trait Function {
    /// Caculates output.
    fn forward(&self, param: &Tensor, input: &Vector) -> Vector;

    /// Returns (gradient of param, new delta)
    fn backward(&self, param: &Tensor, input: &Vector, delta: &Vector) -> (Tensor, Vector);
}

pub struct Linear;

impl Function for Linear {
    fn forward(&self, param: &Tensor, input: &Vector) -> Vector {
        let matrix = param.as_matrix();
        matrix.apply(&input)
    }
    fn backward(&self, param: &Tensor, input: &Vector, delta: &Vector) -> (Tensor, Vector) {
        let mut param_grad: Vec<Vector> = vec![];
        for a in input.0.iter() {
            let mut grad: Vec<f32> = vec![];
            for d in delta.0.iter() {
                grad.push(d * a);
            }
            param_grad.push(Vector(grad));
        }
        let param_grad = Matrix::new(param_grad);

        let matrix = param.as_matrix();
        let matrix_t = matrix.transpose();
        assert_eq!(param_grad.clone().shape(), matrix.shape());
        let delta = matrix_t.apply(&delta);
        (Tensor::M(param_grad), delta)
    }
}

pub struct Translation;

impl Function for Translation {
    fn forward(&self, param: &Tensor, input: &Vector) -> Vector {
        let param = param.as_vector();
        param + (*input).clone()
    }
    fn backward(&self, _param: &Tensor, _input: &Vector, delta: &Vector) -> (Tensor, Vector) {
        (Tensor::V(delta.clone()), delta.clone())
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
    fn forward(&self, _param: &Tensor, input: &Vector) -> Vector {
        Vector(input.0.iter().map(|x| self.eval(*x)).collect())
    }

    fn backward(&self, _param: &Tensor, input: &Vector, delta: &Vector) -> (Tensor, Vector) {
        let derivatives = Vector(input.0.iter().map(|x| self.derivetive(*x)).collect());

        (
            Tensor::N,
            derivatives.hadamard(&delta),
        )
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
        ((*input).clone() - (*target).clone()).len_sq() * 0.5
    }
    fn delta(&self, target: &Vector, input: &Vector) -> Vector {
        (*input).clone() - (*target).clone()
    }
}

pub struct CrossEntropyLoss;

impl LossFunction for CrossEntropyLoss {
    fn loss(&self, target: &Vector, input: &Vector) -> f32 {
        let mut sum = 0.0;
        for i in 0..input.size() {
            let x = input.0[i];
            let y = target.0[i];
            debug_assert!((0.0..=1.0).contains(&x), "input have to be in [0, 1] but got {}", x);
            debug_assert!((0.0..=1.0).contains(&y), "target have to be in [0, 1] but got {}", y);
            let x = x.clamp(1e-5, 1. - 1e-5);
            let y = y.clamp(1e-5, 1. - 1e-5);
            let loss =  -y * x.ln() - (1. - y) * (1. - x).ln();
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
