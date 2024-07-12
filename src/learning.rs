use std::{iter::zip, vec};

use crate::{
    linealg::{Tensor, Vector},
    modules::{Function, LossFunction},
};

pub struct SGD {
    func: Box<dyn Function>,
    param: Tensor,
    loss_fn: Box<dyn LossFunction>,
    loss: f32,
    output_cache: Vector,
    param_grad: Option<Tensor>,
    batch_size: usize,
}

impl SGD {
    pub fn new(func: Box<dyn Function>, param: Tensor, loss_fn: Box<dyn LossFunction>) -> Self {
        Self {
            func,
            param,
            loss_fn,
            loss: 0.0,
            output_cache: Vector(vec![]),
            param_grad: None,
            batch_size: 0,
        }
    }

    pub fn forward(&mut self, input: Vector) {
        self.output_cache = self.func.forward(&self.param, &input);
    }

    /// Only evaluate, returns output on last layer
    pub fn evaluate(&self, input: Vector) -> Vector {
        self.func.evaluate(&self.param, &input)
    }

    pub fn backprop(&mut self, target: Vector, input: Vector) {
        let delta = self.loss_fn.delta(&target, &self.output_cache);
        let (grad, _) = self.func.backward(&self.param, &input, delta);

        if let Some(param_grad_sum) = &mut self.param_grad {
            *param_grad_sum += &grad;
        } else {
            assert!(self.batch_size == 0);
            self.param_grad = Some(grad);
        }

        self.batch_size += 1;
        self.loss += self.loss_fn.loss(&target, &self.output_cache);
    }

    pub fn step(&mut self, lr: f32) {
        if let Some(param_grad) = self.param_grad.take() {
            self.param -= &(param_grad * (lr / self.batch_size as f32));
        } else {
            panic!("No gradient to desent")
        }
    }

    pub fn zero_grad(&mut self) {
        self.param_grad = None;
        self.batch_size = 0;
        self.loss = 0.0;
    }

    pub fn get_param(&self) -> Tensor {
        self.param.clone()
    }

    pub fn set_param(&mut self, param: Tensor) {
        self.param = param;
    }

    /// Returns average loss of last batch
    pub fn get_loss(&self) -> f32 {
        self.loss / self.batch_size as f32
    }
}
