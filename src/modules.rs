use std::{iter::zip, vec};

use rand::distributions::{Bernoulli, Distribution};

use crate::linealg::{Shape, Tensor, Vector};

pub trait Function {
    /// Caculates output.
    fn evaluate(&self, param: &Tensor, input: &Vector) -> Vector;

    /// Evaluate and prepare for backward.
    fn forward(&mut self, param: &Tensor, input: &Vector) -> Vector {
        self.evaluate(param, input)
    }

    /// Returns (gradient of param, new delta)
    fn backward(&self, param: &Tensor, input: &Vector, delta: Vector) -> (Tensor, Vector);

    /// Shape of parameter
    fn param_shape(&self) -> Shape;
}

pub struct Linear {
    from: usize,
    to: usize,
}

impl Linear {
    pub fn new(from: usize, to: usize) -> Self {
        Self { from, to }
    }
}

impl Function for Linear {
    fn evaluate(&self, param: &Tensor, input: &Vector) -> Vector {
        debug_assert!(
            input.0.len() == self.from,
            "Wrong input dimension, expected {} but got {}",
            self.from,
            input.0.len()
        );
        let matrix = param.clone().as_matrix();
        matrix.apply(&input)
    }
    fn backward(&self, param: &Tensor, input: &Vector, delta: Vector) -> (Tensor, Vector) {
        let param_grad = delta.outer(input);
        let matrix = param.clone().as_matrix();
        (Tensor::M(param_grad), matrix.apply_transposed(&delta))
    }
    fn param_shape(&self) -> Shape {
        Shape::M(self.to, self.from)
    }
}

pub struct Translation {
    dim: usize,
}

impl Translation {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Function for Translation {
    fn evaluate(&self, param: &Tensor, input: &Vector) -> Vector {
        debug_assert!(
            input.0.len() == self.dim,
            "Wrong input dimension, expected {} but got {}",
            self.dim,
            input.0.len()
        );
        let param = param.clone().as_vector();
        param + input
    }
    fn backward(&self, _param: &Tensor, _input: &Vector, delta: Vector) -> (Tensor, Vector) {
        (Tensor::V(delta.clone()), delta)
    }
    fn param_shape(&self) -> Shape {
        Shape::V(self.dim)
    }
}

pub struct Softmax {
    // "A vector that got exponetiated"
    exped: Vector,
    expsum: f32,
}

impl Softmax {
    pub fn new() -> Self {
        Self { exped: Vector(vec![]), expsum: 0.0 }
    }
}

impl Function for Softmax {
    fn evaluate(&self, _param: &Tensor, input: &Vector) -> Vector {
        let exped = Vector(input.0.iter().map(|x| x.exp()).collect());
        let expsum = exped.0.iter().sum::<f32>();
        exped * (1. / expsum)
    }

    fn forward(&mut self, _param: &Tensor, input: &Vector) -> Vector {
        self.exped = Vector(input.0.iter().map(|x| x.exp()).collect());
        self.expsum = self.exped.0.iter().sum::<f32>();
        self.exped.clone() * (1. / self.expsum)
    }

    fn backward(&self, _param: &Tensor, _input: &Vector, delta: Vector) -> (Tensor, Vector) {
        let matrix = self.exped.outer(&self.exped) * (-1. / self.expsum.powi(2));
        let second_term = matrix.apply(&delta);
        let first_term = delta.hadamard(&self.exped) * (1. / self.expsum);
        (Tensor::N, first_term + &second_term)
    }

    fn param_shape(&self) -> Shape {
        Shape::N
    }
}

/// Let inner function's parameters be fixed.
pub struct FixParam<T: Function> {
    inner: T,
    param: Tensor,
}

impl<T: Function> FixParam<T> {
    pub fn new(inner: T, param: Tensor) -> Self {
        Self { inner, param }
    }
}

impl<T: Function> Function for FixParam<T> {
    fn evaluate(&self, _: &Tensor, input: &Vector) -> Vector {
        self.inner.evaluate(&self.param, input)
    }

    fn backward(&self, _: &Tensor, input: &Vector, delta: Vector) -> (Tensor, Vector) {
        let (_, delta) = self.inner.backward(&self.param, input, delta);
        (Tensor::N, delta)
    }

    fn param_shape(&self) -> Shape {
        Shape::N
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

    fn param_shape(&self) -> Shape {
        Shape::N
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

    fn param_shape(&self) -> Shape {
        Shape::N
    }
}

pub struct Sine;

impl ActivationFunction for Sine {
    fn eval(&self, input: f32) -> f32 {
        input.sin()
    }
    fn derivetive(&self, input: f32) -> f32 {
        input.cos()
    }
}

pub struct Cosine;

impl ActivationFunction for Cosine {
    fn eval(&self, input: f32) -> f32 {
        input.cos()
    }
    fn derivetive(&self, input: f32) -> f32 {
        -input.sin()
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

    fn param_shape(&self) -> Shape {
        Shape::L(
            self.funcs
                .iter()
                .map(|f| f.param_shape())
                .collect::<Vec<Shape>>(),
        )
    }
}
