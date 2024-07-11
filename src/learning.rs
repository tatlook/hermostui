use std::iter::zip;

use crate::{
    linealg::{Tensor, Vector},
    modules::{Function, LossFunction},
};

pub struct SGD {
    funcs: Vec<Box<dyn Function>>,
    params: Vec<Tensor>,
    loss_fn: Box<dyn LossFunction>,
    loss: f32,
    value_cache: Vec<Vector>,
    param_grads: Option<Vec<Tensor>>,
    batch_size: usize,
}

impl SGD {
    pub fn new(
        funcs: Vec<Box<dyn Function>>,
        params: Vec<Tensor>,
        loss_fn: Box<dyn LossFunction>,
    ) -> Self {
        Self {
            funcs,
            params,
            loss_fn,
            loss: 0.0,
            value_cache: vec![],
            param_grads: None,
            batch_size: 0,
        }
    }

    pub fn forward(&mut self, mut input: Vector) {
        self.value_cache.clear();

        for (func, param) in zip(&mut self.funcs, &self.params) {
            let output = func.forward(param, &input);
            self.value_cache.push(input);
            input = output;
        }
        self.value_cache.push(input);
    }

    /// Only evaluate, returns output on last layer
    pub fn evaluate(&self, mut input: Vector) -> Vector {
        for (func, param) in zip(&self.funcs, &self.params) {
            let output = func.evaluate(param, &input);
            input = output;
        }
        input
    }

    pub fn backprop(&mut self, target: Vector) {
        let mut param_grads = vec![];
        let mut delta = self
            .loss_fn
            .delta(&target, &self.value_cache.last().unwrap());
        for (i, func) in self.funcs.iter().enumerate().rev() {            
            let param = &self.params[i];
            let input = &self.value_cache[i];
            let (grad, new_delta) = func.backward(param, input, delta);
            delta = new_delta;
            param_grads.insert(0, grad);
        }

        if let Some(param_grads_sum) = &mut self.param_grads {
            for (i, grad) in param_grads.iter().enumerate() {
                param_grads_sum[i] += grad;
            }
        } else {
            assert!(self.batch_size == 0);
            self.param_grads = Some(param_grads);
        }

        self.batch_size += 1;
        self.loss += self
            .loss_fn
            .loss(&target, &self.value_cache.last().unwrap());
    }

    pub fn step(&mut self, lr: f32) {
        if let Some(param_grads) = self.param_grads.take() {
            for (i, param_grad) in param_grads.into_iter().enumerate() {
                let param = &mut self.params[i];
                *param -= &(param_grad * (lr / self.batch_size as f32));
            }
        } else {
            panic!("No gradient to desent")
        }
    }

    pub fn zero_grad(&mut self) {
        self.param_grads = None;
        self.batch_size = 0;
        self.loss = 0.0;
    }

    pub fn get_params(&self) -> Vec<Tensor> {
        self.params.clone()
    }

    pub fn set_params(&mut self, params: Vec<Tensor>) {
        self.params = params;
    }

    /// Returns average loss of last batch
    pub fn get_loss(&self) -> f32 {
        self.loss / self.batch_size as f32
    }
}
