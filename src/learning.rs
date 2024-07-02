use std::iter::zip;

use crate::{
    linealg::Tensor,
    modules::{Function, LossFunction},
};

pub struct Sequence {
    input: Tensor,
    funcs: Vec<Box<dyn Function>>,
    params: Vec<Tensor>,
    loss_fn: Box<dyn LossFunction>,
    loss: f32,
    target: Tensor,
    output_chache: Vec<Tensor>,
    param_grads: Option<Vec<Tensor>>,
    batch_size: usize,
}

impl Sequence {
    pub fn new(
        funcs: Vec<Box<dyn Function>>,
        params: Vec<Tensor>,
        loss_fn: Box<dyn LossFunction>,
    ) -> Self {
        Self {
            input: Tensor::N,
            funcs,
            params,
            loss_fn,
            loss: 0.0,
            target: Tensor::N,
            output_chache: vec![],
            param_grads: None,
            batch_size: 0,
        }
    }

    pub fn set_input(&mut self, input: Tensor) {
        self.input = input;
        self.output_chache.clear();
    }

    pub fn set_target(&mut self, target: Tensor) {
        self.target = target;
    }

    pub fn forward(&mut self) {
        self.output_chache.clear();
        let mut input = self.input.clone();

        for (func, param) in zip(&self.funcs, &self.params) {
            let output = func.forward(param, &input);
            self.output_chache.push(output.clone());
            input = output;
        }

        self.loss += self.loss_fn.loss(&self.target, &input);
    }
    
    pub fn backprop(&mut self) {
        let mut param_grads = vec![];
        let mut delta = self
            .loss_fn
            .delta(&self.target, &self.output_chache.last().unwrap());
        for i in 0..self.funcs.len() {
            let i = self.funcs.len() - 1 - i;
            let func = &self.funcs[i];
            let param = &self.params[i];
            let input = if i == 0 {
                &self.input
            } else {
                &self.output_chache[i - 1]
            };
            let (grad, new_delta) = func.backward(param, input, &delta);
            delta = new_delta;
            param_grads.insert(0, grad);
        }
        
        if let Some(param_grads_sum) = &mut self.param_grads {
            for i in 0..param_grads_sum.len() {
                param_grads_sum[i] += param_grads[i].clone();
            }
        } else {
            assert!(self.batch_size == 0);
            self.param_grads = Some(param_grads);
        }
        self.batch_size += 1;
    }
    
    pub fn step(&mut self, lr: f32) {
        if let Some(param_grads) = &self.param_grads {
            for i in 0..self.funcs.len() {
                let param_grad = param_grads[i].clone();
                let param = &mut self.params[i];
                *param -= Tensor::S(lr / self.batch_size as f32) * param_grad;
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
        self.zero_grad();
    }

    /// Returns (network output, loss)
    pub fn get_result(&self) -> (Tensor, f32) {
        let output = self.output_chache.last().unwrap().clone();
        (output, self.loss / self.batch_size as f32)
    }
}
