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
        let matrix = param.as_matrix();
        let input = input.as_vector();
        let matrix_t = matrix.transpose();
        let mut delta = delta.as_vector();
        let mut param_grad: Vec<Vector> = vec![];
        for a in input.0.iter() {
            let mut grad: Vec<f32> = vec![];
            for d in delta.0.iter() {
                grad.push(d * a);
            }
            param_grad.push(Vector(grad));
        }
        assert_eq!(Matrix::new(param_grad.clone()).shape(), matrix.shape());
        delta = matrix_t.apply(&delta);
        (Tensor::M(Matrix::new(param_grad)), Tensor::V(delta))
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

pub struct ReLU;

impl Function for ReLU {
    fn forward(&self, _param: &Tensor, input: &Tensor) -> Tensor {
        let vector = input.as_vector();
        Tensor::V(Vector(vector.0.iter().map(|x| x.max(0.0)).collect()))
    }
    fn backward(&self, _param: &Tensor, input: &Tensor, delta: &Tensor) -> (Tensor, Tensor) {
        let vector = input.as_vector();
        let derivetive = Vector(
            vector
                .0
                .iter()
                .map(|x| if *x >= 0.0 { 1.0 } else { 0.0 })
                .collect(),
        );
        (
            Tensor::N,
            Tensor::V(derivetive.hadamard(&delta.as_vector())),
        )
    }
}

pub struct Sigmoid;

impl Function for Sigmoid {
    fn forward(&self, _param: &Tensor, input: &Tensor) -> Tensor {
        let vector = input.as_vector();
        Tensor::V(Vector(
            vector.0.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(),
        ))
    }
    fn backward(&self, _param: &Tensor, input: &Tensor, delta: &Tensor) -> (Tensor, Tensor) {
        let vector = input.as_vector();
        let derivetive = Vector(
            vector
                .0
                .iter()
                .map(|x| (1. + (-x).exp()).powi(-2) * (-x).exp())
                .collect(),
        );
        (
            Tensor::N,
            Tensor::V(derivetive.hadamard(&delta.as_vector())),
        )
    }
}

pub struct SqLoss;

impl Function for SqLoss {
    fn forward(&self, param: &Tensor, input: &Tensor) -> Tensor {
        Tensor::S((input.as_vector() - param.as_vector()).len_sq() * 0.5)
    }
    fn backward(&self, param: &Tensor, input: &Tensor, _delta: &Tensor) -> (Tensor, Tensor) {
        let y = param.as_vector();
        let x = input.as_vector();
        (
            Tensor::N,
            Tensor::V(y.clone() - x.clone()),
        )
    }
}
