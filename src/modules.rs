use std::vec;

use crate::linealg::{Matrix, Tensor, Vector};

pub trait Function {
    /// Caculates output.
    fn forward(&self, param: &Tensor, input: &Tensor) -> Tensor;

    /// Returns (gradient of param, new delta)
    fn backward(&self, param: &Tensor, input: &Tensor, delta: &Tensor) -> (Tensor, Tensor);
}

pub struct Linear;

impl Function for Linear {
    fn forward(&self, param: &Tensor, input: &Tensor) -> Tensor {
        let matrix = param.as_matrix();
        Tensor::V(matrix.apply(&input.as_vector()))
    }
    fn backward(&self, param: &Tensor, input: &Tensor, delta: &Tensor) -> (Tensor, Tensor) {
        let input = input.as_vector();
        let mut delta = delta.as_vector();

        let mut param_grad: Vec<Vector> = vec![];
        for a in input.0 {
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
        delta = matrix_t.apply(&delta);
        (Tensor::M(param_grad), Tensor::V(delta))
    }
}

pub struct Translation;

impl Function for Translation {
    fn forward(&self, param: &Tensor, input: &Tensor) -> Tensor {
        let param = param.as_vector();
        let input = input.as_vector();
        Tensor::V(param + input)
    }
    fn backward(&self, _param: &Tensor, _input: &Tensor, delta: &Tensor) -> (Tensor, Tensor) {
        (delta.clone(), delta.clone())
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
    fn forward(&self, _param: &Tensor, input: &Tensor) -> Tensor {
        let input = input.as_vector();
        Tensor::V(Vector(input.0.iter().map(|x| self.eval(*x)).collect()))
    }

    fn backward(&self, _param: &Tensor, input: &Tensor, delta: &Tensor) -> (Tensor, Tensor) {
        let input = input.as_vector();
        let derivatives = Vector(input.0.iter().map(|x| self.derivetive(*x)).collect());

        (
            Tensor::N,
            Tensor::V(derivatives.hadamard(&delta.as_vector())),
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
    fn loss(&self, target: &Tensor, input: &Tensor) -> f32;

    fn delta(&self, target: &Tensor, input: &Tensor) -> Tensor;
}

pub struct MSELoss;

impl LossFunction for MSELoss {
    fn loss(&self, target: &Tensor, input: &Tensor) -> f32 {
        (input.as_vector() - target.as_vector()).len_sq() * 0.5
    }
    fn delta(&self, target: &Tensor, input: &Tensor) -> Tensor {
        let y = target.as_vector();
        let x = input.as_vector();
        Tensor::V(x - y)
    }
}
